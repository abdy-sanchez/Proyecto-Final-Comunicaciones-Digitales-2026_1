"""
rx_camera_module.py

Modulo receptor para Fase B del modem optico pantalla-camara.

Cadena implementada:
    captura de una foto con camara
    deteccion de marcadores fiduciales
    estimacion de homografia
    rectificacion
    calibracion de brillo con pilotos
    muestreo de celdas
    demodulacion Manchester/BPSK
    validacion CRC
    correccion limitada de erasures Manchester

Uso tipico en Jupyter:

    import importlib
    import rx_camera_module as rx
    importlib.reload(rx)

    result = rx.rx_from_camera_once(cfg, camera_index=0, debug=True)
    print(result["text"])

Este modulo asume que en el notebook principal ya existen las funciones/constantes
creadas en la Fase A:
    build_role_grid, data_positions, marker_origins,
    PILOT_HIGH, PILOT_LOW, PREAMBLE_BITS, crc16_ccitt,
    y preferiblemente bits_to_bytes.

El modulo las busca automaticamente en __main__ para que el .ipynb quede limpio.
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import sys

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional


# ============================================================
# 0. Enlace con funciones/constantes del transmisor en Jupyter
# ============================================================

_TX_BINDINGS: Dict[str, Any] = {}


def bind_tx_helpers(**kwargs: Any) -> None:
    """
    Permite registrar manualmente funciones/constantes del transmisor.

    Ejemplo opcional en el notebook:
        rx.bind_tx_helpers(
            build_role_grid=build_role_grid,
            data_positions=data_positions,
            marker_origins=marker_origins,
            PILOT_HIGH=PILOT_HIGH,
            PILOT_LOW=PILOT_LOW,
            PREAMBLE_BITS=PREAMBLE_BITS,
            crc16_ccitt=crc16_ccitt,
            bits_to_bytes=bits_to_bytes,
        )

    Si no se llama, el modulo intenta encontrarlas en __main__.
    """
    _TX_BINDINGS.update(kwargs)


def _tx(name: str) -> Any:
    """
    Busca una funcion/constante del transmisor.
    Primero revisa bindings manuales y luego el namespace principal del notebook.
    """
    if name in _TX_BINDINGS:
        return _TX_BINDINGS[name]

    main = sys.modules.get("__main__")
    if main is not None and hasattr(main, name):
        return getattr(main, name)

    raise NameError(
        f"No se encontro '{name}'. Ejecuta primero las celdas del transmisor "
        f"o registra dependencias con rx.bind_tx_helpers(...)."
    )


# ============================================================
# 1. Patrones de marcadores
# ============================================================

STANDARD_FINDER_PATTERN = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
], dtype=np.uint8)


# Marcador anchor recomendado en la ultima version.
ANCHOR_PATTERN_A = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
], dtype=np.uint8)


# Marcador anchor alternativo, por compatibilidad con una version anterior.
ANCHOR_PATTERN_B = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1],
    [1, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
], dtype=np.uint8)


@dataclass
class MarkerCandidate:
    center: Tuple[float, float]
    quad: np.ndarray
    area: float
    marker_type: str
    pattern_error: int
    rotation_k: int
    sampled_pattern: np.ndarray


# ============================================================
# 2. Utilidades generales
# ============================================================

def _as_gray(img: np.ndarray) -> np.ndarray:
    """Convierte BGR/gris a gris uint8."""
    if img.ndim == 2:
        return img.astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Ordena 4 puntos aproximadamente como:
    top-left, top-right, bottom-right, bottom-left.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    rect = np.zeros((4, 2), dtype=np.float32)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect


def _bits_to_int(bits: np.ndarray) -> int:
    """Bits MSB-first a entero."""
    bits = np.asarray(bits).astype(int).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def _bits_to_bytes_rx(bits: np.ndarray) -> bytes:
    """
    Convierte bits a bytes.
    Usa bits_to_bytes del transmisor si existe, para mantener compatibilidad.
    """
    bits = np.asarray(bits, dtype=np.uint8).flatten()

    try:
        bits_to_bytes = _tx("bits_to_bytes")
        out = bits_to_bytes(bits)
        return bytes(out)
    except NameError:
        pass

    pad = (-bits.size) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

    return np.packbits(bits).tobytes()


def _ideal_marker_centers(cfg: Any, rectified_cell_size: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Centros ideales de los marcadores dentro de la grilla rectificada.
    Retorna puntos en pixeles de la imagen rectificada.
    """
    marker_origins = _tx("marker_origins")
    origins = list(marker_origins(cfg))

    centers = []
    for r0, c0 in origins:
        x = (c0 + 3.5) * rectified_cell_size
        y = (r0 + 3.5) * rectified_cell_size
        centers.append([x, y])

    return np.asarray(centers, dtype=np.float32), origins


# ============================================================
# 3. Captura de una sola foto con camara
# ============================================================

def capture_one_photo(
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    warmup_frames: int = 20,
    window_name: str = "RX - presiona ESPACIO para capturar",
) -> np.ndarray:
    """
    Abre la camara, muestra vista previa y captura una sola foto.

    Controles:
        ESPACIO / ENTER / s / c  -> capturar
        q / ESC                  -> cancelar
    """
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la camara con indice {camera_index}. "
            "Prueba con camera_index=1 o revisa permisos de camara."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # No todas las camaras aceptan estas opciones.
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    try:
        for _ in range(warmup_frames):
            cap.read()
            time.sleep(0.02)

        captured = None

        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("La camara no entrego frame.")

            preview = frame.copy()
            cv2.putText(
                preview,
                "ESPACIO/ENTER: capturar | q/ESC: cancelar",
                (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (13, 32, ord("s"), ord("c")):
                captured = frame.copy()
                break

            if key in (27, ord("q")):
                raise RuntimeError("Captura cancelada por el usuario.")

        return captured

    finally:
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(3):
            cv2.waitKey(1)


# ============================================================
# 4. Deteccion de marcadores fiduciales
# ============================================================

def _sample_7x7_pattern(marker_gray: np.ndarray, margin_frac: float = 0.25) -> np.ndarray:
    """
    Toma un marcador rectificado y estima su patron binario 7x7.
    """
    marker_gray = _as_gray(marker_gray)
    h, w = marker_gray.shape

    cell_h = h / 7.0
    cell_w = w / 7.0

    means = np.zeros((7, 7), dtype=np.float32)

    for r in range(7):
        for c in range(7):
            y0 = int((r + margin_frac) * cell_h)
            y1 = int((r + 1 - margin_frac) * cell_h)
            x0 = int((c + margin_frac) * cell_w)
            x1 = int((c + 1 - margin_frac) * cell_w)

            roi = marker_gray[y0:y1, x0:x1]
            means[r, c] = float(np.median(roi)) if roi.size else 0.0

    lo = np.percentile(means, 15)
    hi = np.percentile(means, 85)
    threshold = 0.5 * (lo + hi)

    return (means >= threshold).astype(np.uint8)


def _classify_finder_pattern(sampled_pattern: np.ndarray) -> Tuple[str, int, int]:
    """
    Compara un patron 7x7 contra los patrones conocidos.
    Retorna:
        marker_type, error_hamming, rotacion_k
    """
    templates = [
        ("standard", STANDARD_FINDER_PATTERN),
        ("anchor", ANCHOR_PATTERN_A),
        ("anchor", ANCHOR_PATTERN_B),
    ]

    best_type = "unknown"
    best_error = 10**9
    best_k = 0

    for marker_type, template in templates:
        for k in range(4):
            rotated = np.rot90(template, k)
            err = int(np.count_nonzero(sampled_pattern != rotated))

            if err < best_error:
                best_error = err
                best_type = marker_type
                best_k = k

    return best_type, best_error, best_k


def _warp_marker_candidate(gray: np.ndarray, quad: np.ndarray, out_px: int = 140) -> np.ndarray:
    """Rectifica un candidato a marcador a una imagen cuadrada."""
    dst = np.array([
        [0, 0],
        [out_px - 1, 0],
        [out_px - 1, out_px - 1],
        [0, out_px - 1],
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(gray, H, (out_px, out_px))


def detect_finder_markers(
    frame: np.ndarray,
    max_pattern_errors: int = 9,
    max_return: int = 12,
    debug: bool = False,
) -> List[MarkerCandidate]:
    """
    Detecta candidatos a marcadores fiduciales en la imagen capturada.
    """
    gray = _as_gray(frame)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    _, bw = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        bw,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    img_area = gray.shape[0] * gray.shape[1]
    min_area = 0.00015 * img_area
    max_area = 0.20 * img_area

    candidates: List[MarkerCandidate] = []

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)

        if len(approx) != 4:
            continue

        if not cv2.isContourConvex(approx):
            continue

        quad = _order_quad_points(approx.reshape(4, 2))

        side_a = np.linalg.norm(quad[0] - quad[1])
        side_b = np.linalg.norm(quad[1] - quad[2])
        if side_a < 10 or side_b < 10:
            continue

        aspect = max(side_a, side_b) / max(1e-6, min(side_a, side_b))
        if aspect > 2.2:
            continue

        marker_warp = _warp_marker_candidate(gray_eq, quad, out_px=140)
        sampled = _sample_7x7_pattern(marker_warp)
        marker_type, err, rot_k = _classify_finder_pattern(sampled)

        if err > max_pattern_errors:
            continue

        center = tuple(np.mean(quad, axis=0).astype(float))

        cand = MarkerCandidate(
            center=center,
            quad=quad,
            area=area,
            marker_type=marker_type,
            pattern_error=err,
            rotation_k=rot_k,
            sampled_pattern=sampled,
        )
        candidates.append(cand)

    candidates.sort(
        key=lambda c: (
            c.pattern_error,
            0 if c.marker_type == "anchor" else 1,
            -c.area,
        )
    )

    # Supresion simple de duplicados por cercania.
    selected: List[MarkerCandidate] = []
    for cand in candidates:
        cx, cy = cand.center
        duplicate = False

        for prev in selected:
            px, py = prev.center
            dist = np.hypot(cx - px, cy - py)
            size_ref = 0.5 * (np.sqrt(cand.area) + np.sqrt(prev.area))

            if dist < 0.45 * size_ref:
                duplicate = True
                break

        if not duplicate:
            selected.append(cand)

        if len(selected) >= max_return:
            break

    if debug:
        print(f"Marcadores candidatos detectados: {len(selected)}")
        for i, c in enumerate(selected):
            print(
                f"  {i}: tipo={c.marker_type}, "
                f"error={c.pattern_error}, "
                f"centro=({c.center[0]:.1f}, {c.center[1]:.1f}), "
                f"area={c.area:.1f}"
            )

    return selected


def draw_marker_overlay(frame: np.ndarray, markers: List[MarkerCandidate]) -> np.ndarray:
    """Dibuja los marcadores detectados sobre la imagen."""
    out = frame.copy()

    for i, m in enumerate(markers):
        quad = m.quad.astype(int)
        cv2.polylines(out, [quad], True, (0, 255, 0), 2)

        cx, cy = map(int, m.center)
        label = f"{i}:{m.marker_type}, e={m.pattern_error}"
        cv2.putText(
            out,
            label,
            (cx - 40, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return out


# ============================================================
# 5. Calibracion de brillo y muestreo de celdas
# ============================================================

def _sample_cell_center(
    gray: np.ndarray,
    row: int,
    col: int,
    cell_px: int,
    margin_frac: float = 0.30,
) -> float:
    """
    Muestra una celda usando solo su region central.
    Usa mediana para reducir reflejos puntuales.
    """
    y0 = int((row + margin_frac) * cell_px)
    y1 = int((row + 1 - margin_frac) * cell_px)
    x0 = int((col + margin_frac) * cell_px)
    x1 = int((col + 1 - margin_frac) * cell_px)

    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = min(gray.shape[0], y1)
    x1 = min(gray.shape[1], x1)

    roi = gray[y0:y1, x0:x1]

    if roi.size == 0:
        return float("nan")

    return float(np.median(roi))


def calibrate_and_sample_symbols(
    warped_grid: np.ndarray,
    cfg: Any,
    rectified_cell_size: Optional[int] = None,
    margin_frac: float = 0.30,
) -> Dict[str, Any]:
    """
    Calibra brillo usando pilotos y muestrea todas las celdas DATA.
    """
    cell_px = int(rectified_cell_size or cfg.cell_size)
    gray = _as_gray(warped_grid)

    build_role_grid = _tx("build_role_grid")
    data_positions = _tx("data_positions")
    PILOT_HIGH = _tx("PILOT_HIGH")
    PILOT_LOW = _tx("PILOT_LOW")

    roles = build_role_grid(cfg)

    high_vals = []
    low_vals = []

    for r in range(cfg.grid_rows):
        for c in range(cfg.grid_cols):
            role = roles[r, c]
            val = _sample_cell_center(gray, r, c, cell_px, margin_frac)

            if np.isnan(val):
                continue

            if role == PILOT_HIGH:
                high_vals.append(val)
            elif role == PILOT_LOW:
                low_vals.append(val)

    positions = data_positions(cfg)

    data_values = []
    for r, c in positions:
        val = _sample_cell_center(gray, r, c, cell_px, margin_frac)
        data_values.append(val)

    data_values = np.asarray(data_values, dtype=np.float32)

    if len(high_vals) >= 1 and len(low_vals) >= 1:
        high_level = float(np.median(high_vals))
        low_level = float(np.median(low_vals))
    else:
        valid = data_values[np.isfinite(data_values)]
        low_level = float(np.percentile(valid, 10))
        high_level = float(np.percentile(valid, 90))

    if high_level < low_level:
        high_level, low_level = low_level, high_level

    span = max(1.0, high_level - low_level)

    normalized = (data_values - low_level) / span
    normalized = np.clip(normalized, 0.0, 1.0)

    symbols = (normalized >= 0.5).astype(np.uint8)
    confidence = np.clip(np.abs(normalized - 0.5) * 2.0, 0.0, 1.0)

    calibrated_gray = np.clip((gray.astype(np.float32) - low_level) * 255.0 / span, 0, 255)
    calibrated_gray = calibrated_gray.astype(np.uint8)

    return {
        "symbols": symbols,
        "values": data_values,
        "normalized": normalized,
        "confidence": confidence,
        "low_level": low_level,
        "high_level": high_level,
        "calibrated_gray": calibrated_gray,
        "data_positions": positions,
    }


# ============================================================
# 6. Manchester/BPSK + CRC + correccion limitada de erasures
# ============================================================

def manchester_decode_symbols(
    symbols: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    convention: str = "01->0",
) -> Dict[str, Any]:
    """
    Decodifica simbolos Manchester.

    convention:
        "01->0": 01 representa bit 0, 10 representa bit 1.
        "01->1": 01 representa bit 1, 10 representa bit 0.

    Pares 00 y 11 se marcan como erasures.
    """
    symbols = np.asarray(symbols, dtype=np.uint8).flatten()

    n_pairs = symbols.size // 2
    raw_bits = np.full(n_pairs, -1, dtype=np.int16)

    invalid_pairs = 0
    erasure_positions = []

    for i in range(n_pairs):
        a = int(symbols[2 * i])
        b = int(symbols[2 * i + 1])

        if convention == "01->0":
            if a == 0 and b == 1:
                raw_bits[i] = 0
            elif a == 1 and b == 0:
                raw_bits[i] = 1
            else:
                invalid_pairs += 1
                erasure_positions.append(i)

        elif convention == "01->1":
            if a == 0 and b == 1:
                raw_bits[i] = 1
            elif a == 1 and b == 0:
                raw_bits[i] = 0
            else:
                invalid_pairs += 1
                erasure_positions.append(i)
        else:
            raise ValueError("convention debe ser '01->0' o '01->1'.")

    return {
        "raw_bits": raw_bits,
        "erasure_positions": np.asarray(erasure_positions, dtype=int),
        "invalid_pairs": invalid_pairs,
        "n_pairs": n_pairs,
        "convention": convention,
    }


def _parse_packet_bits(
    bits: np.ndarray,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Interpreta:
        PREAMBLE_BITS           32
        frame_index             16
        total_frames            16
        payload_chunk_length    16
        payload_chunk_bits       N
        crc16                   16
    """
    bits = np.asarray(bits, dtype=np.uint8).flatten()

    PREAMBLE_BITS = np.asarray(_tx("PREAMBLE_BITS"), dtype=np.uint8).flatten()
    crc16_ccitt = _tx("crc16_ccitt")

    header_len = 32 + 16 + 16 + 16

    result = {
        "crc_ok": False,
        "text": "",
        "payload_bits": np.array([], dtype=np.uint8),
        "frame_index": None,
        "total_frames": None,
        "payload_len_bits": None,
        "packet_len_bits": None,
        "preamble_errors": None,
        "rx_crc": None,
        "calc_crc": None,
        "structure_ok": False,
        "error": None,
        "score": 10**9,
    }

    if bits.size < header_len + 16:
        result["error"] = "No hay suficientes bits para header + CRC."
        return result

    preamble_errors = int(np.count_nonzero(bits[:32] != PREAMBLE_BITS))

    frame_index = _bits_to_int(bits[32:48])
    total_frames = _bits_to_int(bits[48:64])
    payload_len = _bits_to_int(bits[64:80])

    max_payload_len = bits.size - header_len - 16
    packet_len = header_len + payload_len + 16

    result.update({
        "frame_index": frame_index,
        "total_frames": total_frames,
        "payload_len_bits": payload_len,
        "packet_len_bits": packet_len,
        "preamble_errors": preamble_errors,
    })

    if payload_len < 0 or payload_len > max_payload_len:
        result["error"] = (
            f"Longitud de payload invalida: {payload_len} bits. "
            f"Maximo posible: {max_payload_len} bits."
        )
        result["score"] = 5000 + preamble_errors * 50
        return result

    payload_start = header_len
    payload_end = payload_start + payload_len
    crc_start = payload_end
    crc_end = crc_start + 16

    protected_bits = bits[:payload_end]
    payload_bits = bits[payload_start:payload_end]
    rx_crc = _bits_to_int(bits[crc_start:crc_end])
    calc_crc = crc16_ccitt(_bits_to_bytes_rx(protected_bits))

    crc_ok = (rx_crc == calc_crc) and (preamble_errors == 0)

    payload_bytes = _bits_to_bytes_rx(payload_bits)
    n_payload_bytes = payload_len // 8
    payload_bytes_for_text = payload_bytes[:n_payload_bytes]

    try:
        text = payload_bytes_for_text.decode(encoding, errors="replace")
    except Exception:
        text = ""

    result.update({
        "crc_ok": bool(crc_ok),
        "text": text,
        "payload_bits": payload_bits,
        "rx_crc": rx_crc,
        "calc_crc": calc_crc,
        "structure_ok": True,
        "error": None if crc_ok else "CRC invalido o preambulo invalido.",
        "score": preamble_errors * 50 + (0 if crc_ok else 800),
    })

    return result


def parse_packet_with_erasure_correction(
    raw_bits_with_erasures: np.ndarray,
    erasure_positions: np.ndarray,
    max_erasures: int = 12,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Correccion limitada:
    - Los pares Manchester invalidos 00/11 se tratan como erasures.
    - Si hay pocos erasures, se prueban combinaciones hasta que el CRC coincida.

    No reemplaza un FEC real tipo Reed-Solomon.
    """
    raw = np.asarray(raw_bits_with_erasures, dtype=np.int16).flatten()
    erasures = np.asarray(erasure_positions, dtype=int).flatten()

    if erasures.size == 0:
        parsed = _parse_packet_bits(raw.astype(np.uint8), encoding=encoding)
        parsed["erasure_correction_used"] = False
        parsed["corrected_erasures"] = 0
        return parsed

    fallback = raw.copy()
    fallback[fallback < 0] = 0
    best = _parse_packet_bits(fallback.astype(np.uint8), encoding=encoding)
    best["erasure_correction_used"] = False
    best["corrected_erasures"] = 0

    if erasures.size > max_erasures:
        best["error"] = (
            f"Demasiados erasures Manchester ({erasures.size}). "
            f"Maximo configurado: {max_erasures}."
        )
        best["score"] += erasures.size * 25
        return best

    for combo in itertools.product([0, 1], repeat=erasures.size):
        candidate = raw.copy()
        candidate[erasures] = np.asarray(combo, dtype=np.int16)

        parsed = _parse_packet_bits(candidate.astype(np.uint8), encoding=encoding)

        if parsed["crc_ok"]:
            parsed["erasure_correction_used"] = True
            parsed["corrected_erasures"] = int(erasures.size)
            return parsed

        if parsed["score"] < best["score"]:
            best = parsed
            best["erasure_correction_used"] = False
            best["corrected_erasures"] = 0

    return best


def decode_warped_grid(
    warped_grid: np.ndarray,
    cfg: Any,
    rectified_cell_size: Optional[int] = None,
    max_erasures: int = 0,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Decodifica una grilla ya rectificada.
    Prueba ambas convenciones Manchester y escoge la que valide CRC.
    """
    sampled = calibrate_and_sample_symbols(
        warped_grid,
        cfg,
        rectified_cell_size=rectified_cell_size,
    )

    results = []

    for convention in ("01->0", "01->1"):
        man = manchester_decode_symbols(
            sampled["symbols"],
            sampled["confidence"],
            convention=convention,
        )

        parsed = parse_packet_with_erasure_correction(
            man["raw_bits"],
            man["erasure_positions"],
            max_erasures=max_erasures,
            encoding=encoding,
        )

        score = (
            parsed["score"]
            + man["invalid_pairs"] * 20
            + int(np.mean(1.0 - sampled["confidence"]) * 100)
        )

        combined = {
            **parsed,
            "score": score,
            "manchester": man,
            "sampled": sampled,
            "convention": convention,
            "invalid_manchester_pairs": man["invalid_pairs"],
        }

        results.append(combined)

    valid = [r for r in results if r["crc_ok"]]

    if valid:
        return min(valid, key=lambda r: r["score"])

    return min(results, key=lambda r: r["score"])


# ============================================================
# 7. Homografia, rectificacion y busqueda de orientacion
# ============================================================

def _generate_marker_combos(
    markers: List[MarkerCandidate],
    max_marker_candidates: int = 8,
    enforce_anchor: bool = True,
):
    """Genera combinaciones razonables de 4 marcadores."""
    n = min(len(markers), max_marker_candidates)
    idxs = list(range(n))

    combos = list(itertools.combinations(idxs, 4))

    if enforce_anchor:
        combos = [
            combo for combo in combos
            if any(markers[i].marker_type == "anchor" for i in combo)
        ]

    return combos


def process_captured_frame(
    frame: np.ndarray,
    cfg: Any,
    rectified_cell_size: Optional[int] = None,
    max_marker_candidates: int = 8,
    final_erasures: int = 12,
    debug: bool = True,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Procesa una foto capturada:
        deteccion de marcadores
        homografia
        rectificacion
        calibracion
        muestreo
        Manchester/BPSK
        CRC y correccion limitada de erasures
    """
    cell_px = int(rectified_cell_size or cfg.cell_size)

    dst_w = cfg.grid_cols * cell_px
    dst_h = cfg.grid_rows * cell_px

    markers = detect_finder_markers(frame, debug=debug)
    overlay = draw_marker_overlay(frame, markers)

    if len(markers) < 4:
        return {
            "ok": False,
            "error": f"Se detectaron {len(markers)} marcadores. Se requieren al menos 4.",
            "text": "",
            "captured": frame,
            "overlay": overlay,
            "markers": markers,
            "warped": None,
            "decoded": None,
        }

    ideal_centers, ideal_origins = _ideal_marker_centers(cfg, cell_px)
    top_left_origin = min(ideal_origins, key=lambda rc: (rc[0], rc[1]))

    evaluations = []

    anchor_exists = any(m.marker_type == "anchor" for m in markers)
    passes = [True, False] if anchor_exists else [False]

    for enforce_anchor in passes:
        combos = _generate_marker_combos(
            markers,
            max_marker_candidates=max_marker_candidates,
            enforce_anchor=enforce_anchor,
        )

        for combo in combos:
            combo_markers = [markers[i] for i in combo]
            src = np.asarray([m.center for m in combo_markers], dtype=np.float32)

            anchor_positions = [
                i for i, m in enumerate(combo_markers)
                if m.marker_type == "anchor"
            ]

            for perm in itertools.permutations(range(4)):
                if enforce_anchor and anchor_positions:
                    anchor_pos = anchor_positions[0]
                    assigned_origin = ideal_origins[perm[anchor_pos]]

                    if assigned_origin != top_left_origin:
                        continue

                dst = ideal_centers[list(perm)].astype(np.float32)

                try:
                    H = cv2.getPerspectiveTransform(src, dst)
                except cv2.error:
                    continue

                warped = cv2.warpPerspective(frame, H, (dst_w, dst_h))

                decoded = decode_warped_grid(
                    warped,
                    cfg,
                    rectified_cell_size=cell_px,
                    max_erasures=0,
                    encoding=encoding,
                )

                marker_penalty = sum(m.pattern_error for m in combo_markers) * 3
                score = decoded["score"] + marker_penalty

                evaluations.append({
                    "score": score,
                    "H": H,
                    "combo": combo,
                    "perm": perm,
                    "decoded": decoded,
                    "enforce_anchor": enforce_anchor,
                })

                if decoded["crc_ok"]:
                    return {
                        "ok": True,
                        "error": None,
                        "text": decoded["text"],
                        "captured": frame,
                        "overlay": overlay,
                        "markers": markers,
                        "warped": warped,
                        "decoded": decoded,
                        "H": H,
                        "combo": combo,
                        "perm": perm,
                    }

    if not evaluations:
        return {
            "ok": False,
            "error": "No se pudo construir una homografia valida.",
            "text": "",
            "captured": frame,
            "overlay": overlay,
            "markers": markers,
            "warped": None,
            "decoded": None,
        }

    # Reintento final con correccion de erasures sobre las mejores geometrias.
    evaluations.sort(key=lambda ev: ev["score"])

    best_final = None
    best_warped = None
    best_ev = None

    for ev in evaluations[:10]:
        H = ev["H"]
        warped = cv2.warpPerspective(frame, H, (dst_w, dst_h))

        decoded = decode_warped_grid(
            warped,
            cfg,
            rectified_cell_size=cell_px,
            max_erasures=final_erasures,
            encoding=encoding,
        )

        if best_final is None or decoded["score"] < best_final["score"]:
            best_final = decoded
            best_warped = warped
            best_ev = ev

        if decoded["crc_ok"]:
            return {
                "ok": True,
                "error": None,
                "text": decoded["text"],
                "captured": frame,
                "overlay": overlay,
                "markers": markers,
                "warped": warped,
                "decoded": decoded,
                "H": H,
                "combo": ev["combo"],
                "perm": ev["perm"],
            }

    return {
        "ok": False,
        "error": "No se logro validar CRC. Revisa enfoque, exposicion, distancia o deteccion de marcadores.",
        "text": best_final["text"] if best_final else "",
        "captured": frame,
        "overlay": overlay,
        "markers": markers,
        "warped": best_warped,
        "decoded": best_final,
        "H": best_ev["H"] if best_ev else None,
        "combo": best_ev["combo"] if best_ev else None,
        "perm": best_ev["perm"] if best_ev else None,
    }


# ============================================================
# 8. Visualizacion de diagnostico
# ============================================================

def show_rx_result(result: Dict[str, Any]) -> None:
    """Muestra diagnostico visual del receptor."""
    captured = result.get("captured")
    overlay = result.get("overlay")
    warped = result.get("warped")
    decoded = result.get("decoded")

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    if captured is not None:
        if captured.ndim == 3:
            plt.imshow(cv2.cvtColor(captured, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(captured, cmap="gray", vmin=0, vmax=255)
    plt.title("Foto capturada")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    if overlay is not None:
        if overlay.ndim == 3:
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(overlay, cmap="gray", vmin=0, vmax=255)
    plt.title("Marcadores detectados")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    if warped is not None:
        plt.imshow(_as_gray(warped), cmap="gray", vmin=0, vmax=255)
    plt.title("Grilla rectificada")
    plt.axis("off")

    plt.show()

    if decoded is not None:
        sampled = decoded.get("sampled", {})
        print("========== Diagnostico RX ==========")
        print(f"OK CRC: {decoded.get('crc_ok')}")
        print(f"Convencion Manchester: {decoded.get('convention')}")
        print(f"Pares Manchester invalidos: {decoded.get('invalid_manchester_pairs')}")
        print(f"Nivel bajo estimado: {sampled.get('low_level')}")
        print(f"Nivel alto estimado: {sampled.get('high_level')}")
        print(f"Errores de preambulo: {decoded.get('preamble_errors')}")
        print(f"Frame index: {decoded.get('frame_index')}")
        print(f"Total frames: {decoded.get('total_frames')}")
        print(f"Payload bits: {decoded.get('payload_len_bits')}")
        print(f"CRC RX: {decoded.get('rx_crc')}")
        print(f"CRC calculado: {decoded.get('calc_crc')}")
        print(f"Correccion erasures usada: {decoded.get('erasure_correction_used')}")
        print(f"Erasures corregidos: {decoded.get('corrected_erasures')}")
        print("Texto decodificado:")
        print(decoded.get("text", ""))


# ============================================================
# 9. Funcion principal: foto unica desde camara y decodificacion
# ============================================================

def rx_from_camera_once(
    cfg: Any,
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    rectified_cell_size: Optional[int] = None,
    debug: bool = True,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Captura una foto con la camara y ejecuta todo el receptor.
    """
    frame = capture_one_photo(
        camera_index=camera_index,
        width=width,
        height=height,
    )

    result = process_captured_frame(
        frame,
        cfg,
        rectified_cell_size=rectified_cell_size,
        debug=debug,
        encoding=encoding,
    )

    if debug:
        show_rx_result(result)

    if result["ok"]:
        print("Decodificacion exitosa.")
    else:
        print("Decodificacion fallida.")
        print(result.get("error"))

    return result


# ============================================================
# 10. Funcion auxiliar para probar con una imagen guardada
# ============================================================

def rx_from_image_file(
    path: str,
    cfg: Any,
    rectified_cell_size: Optional[int] = None,
    debug: bool = True,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Util para probar con una foto guardada antes de usar camara en vivo.
    """
    frame = cv2.imread(path, cv2.IMREAD_COLOR)

    if frame is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")

    result = process_captured_frame(
        frame,
        cfg,
        rectified_cell_size=rectified_cell_size,
        debug=debug,
        encoding=encoding,
    )

    if debug:
        show_rx_result(result)

    return result


__all__ = [
    "bind_tx_helpers",
    "capture_one_photo",
    "detect_finder_markers",
    "draw_marker_overlay",
    "calibrate_and_sample_symbols",
    "manchester_decode_symbols",
    "decode_warped_grid",
    "process_captured_frame",
    "show_rx_result",
    "rx_from_camera_once",
    "rx_from_image_file",
    "MarkerCandidate",
]
