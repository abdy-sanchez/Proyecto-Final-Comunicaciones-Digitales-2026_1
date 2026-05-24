"""
rx_camera_module_fase_c.py

Receptor Fase C para el modem optico pantalla-camara.

Este modulo conserva la cadena robusta de Fase B:
    deteccion de marcadores fiduciales
    estimacion de homografia
    rectificacion
    calibracion de brillo con pilotos
    muestreo de celdas
    demodulacion Manchester/BPSK
    validacion CRC por frame

y agrega la capa de recepcion continua para Fase C:
    captura continua de video
    deteccion de SYNC
    lectura de DATA frames con numero de secuencia
    descarte de duplicados
    seleccion de la lectura mas confiable por secuencia
    validacion CRC por frame
    delimitador END
    reconstruccion y CRC global del mensaje

Uso tipico en Jupyter:

    import importlib
    import modulo_1_fase_c_tx as tx
    import rx_camera_module_fase_c as rx

    importlib.reload(tx)
    importlib.reload(rx)

    cfg = tx.TxVisualConfig(frame_duration_s=0.12)
    rx.bind_tx_module(tx)     # recomendado

    result = rx.rx_from_camera_continuous(
        cfg,
        camera_index=0,
        max_seconds=15,
        require_sync=True,
        require_end=True,
        show_preview=True,
    )

    print(result["text"])

El modulo tambien intenta encontrar automaticamente funciones/constantes en
__main__ o en modulo_1_fase_c_tx si no se llama bind_tx_module(...).
"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import sys
import importlib
import math

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Sequence


# ============================================================
# 0. Enlace con funciones/constantes del transmisor en Jupyter
# ============================================================

_TX_BINDINGS: Dict[str, Any] = {}
_TX_MODULE: Optional[Any] = None


def bind_tx_helpers(**kwargs: Any) -> None:
    """
    Permite registrar manualmente funciones/constantes del transmisor.

    Ejemplo opcional en el notebook:
        rx.bind_tx_helpers(
            build_role_grid=tx.build_role_grid,
            data_positions=tx.data_positions,
            marker_origins=tx.marker_origins,
            PILOT_HIGH=tx.PILOT_HIGH,
            PILOT_LOW=tx.PILOT_LOW,
            PREAMBLE_BITS=tx.PREAMBLE_BITS,
            crc16_ccitt=tx.crc16_ccitt,
            bits_to_bytes=tx.bits_to_bytes,
        )
    """
    _TX_BINDINGS.update(kwargs)


def bind_tx_module(tx_module: Any) -> None:
    """
    Registra el modulo transmisor completo.

    Uso recomendado:
        import modulo_1_fase_c_tx as tx
        import rx_camera_module_fase_c as rx
        rx.bind_tx_module(tx)
    """
    global _TX_MODULE
    _TX_MODULE = tx_module


def _tx(name: str) -> Any:
    """
    Busca una funcion/constante del transmisor.
    Orden:
        1) bindings manuales
        2) modulo registrado con bind_tx_module(...)
        3) namespace principal del notebook
        4) import automatico de modulo_1_fase_c_tx
    """
    if name in _TX_BINDINGS:
        return _TX_BINDINGS[name]

    if _TX_MODULE is not None and hasattr(_TX_MODULE, name):
        return getattr(_TX_MODULE, name)

    main = sys.modules.get("__main__")
    if main is not None and hasattr(main, name):
        return getattr(main, name)

    try:
        tx_mod = importlib.import_module("modulo_1_fase_c_tx")
        if hasattr(tx_mod, name):
            return getattr(tx_mod, name)
    except Exception:
        pass

    raise NameError(
        f"No se encontro '{name}'. Importa el transmisor Fase C y registra sus "
        f"dependencias con rx.bind_tx_module(tx) o rx.bind_tx_helpers(...)."
    )


# ============================================================
# 0.1 Constantes del protocolo Fase C
# ============================================================

# Valores definidos en modulo_1_fase_c_tx.py. Se duplican aqui como fallback
# para que el receptor pueda interpretar el header aun si no se importan.
FRAME_TYPE_SYNC = 1
FRAME_TYPE_DATA = 2
FRAME_TYPE_END = 3

FRAME_TYPE_NAMES = {
    FRAME_TYPE_SYNC: "SYNC",
    FRAME_TYPE_DATA: "DATA",
    FRAME_TYPE_END: "END",
}

PROTOCOL_VERSION = 1
HEADER_BITS_NO_PAYLOAD = 32 + 8 + 8 + 16 + 16 + 16 + 16 + 16
FRAME_CRC_BITS = 16


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


def _safe_tx_constant(name: str, fallback: Any) -> Any:
    """Retorna una constante del TX si existe; si no, usa fallback."""
    try:
        return _tx(name)
    except NameError:
        return fallback


def _estimate_frame_confidence(decoded: Dict[str, Any]) -> float:
    """
    Metrica simple de confianza para comparar lecturas duplicadas del mismo SEQ.
    Solo se usa para decidir si reemplazar un DATA ya recibido.
    """
    sampled = decoded.get("sampled", {}) if isinstance(decoded, dict) else {}
    conf_arr = sampled.get("confidence")

    if conf_arr is None:
        base = 0.0
    else:
        arr = np.asarray(conf_arr, dtype=np.float32)
        base = float(np.nanmean(arr)) if arr.size else 0.0

    invalid = int(decoded.get("invalid_manchester_pairs") or 0)
    n_pairs = max(1, int(decoded.get("manchester", {}).get("n_pairs", 1)))
    preamble_errors = int(decoded.get("preamble_errors") or 0)

    penalty = 0.25 * (invalid / n_pairs) + 0.03 * preamble_errors
    if not decoded.get("crc_ok", False):
        penalty += 0.5

    return float(np.clip(base - penalty, 0.0, 1.0))


def _parse_packet_bits(
    bits: np.ndarray,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Interpreta el paquete Fase C:
        PREAMBLE_BITS          32
        protocol_version        8
        frame_type              8   1=SYNC, 2=DATA, 3=END
        sequence               16
        total_data_frames      16
        payload_len_bits       16
        message_len_bytes      16
        message_crc16          16
        payload_bits            N
        frame_crc16            16

    El CRC del frame protege todo desde el preambulo hasta el payload.
    El CRC global del mensaje se verifica al reconstruir todos los DATA frames.
    """
    bits = np.asarray(bits, dtype=np.uint8).flatten()

    PREAMBLE_BITS = np.asarray(_tx("PREAMBLE_BITS"), dtype=np.uint8).flatten()
    crc16_ccitt = _tx("crc16_ccitt")

    expected_protocol_version = int(_safe_tx_constant("PROTOCOL_VERSION", PROTOCOL_VERSION))
    header_len = int(_safe_tx_constant("HEADER_BITS_NO_PAYLOAD", HEADER_BITS_NO_PAYLOAD))

    result = {
        "crc_ok": False,
        "text": "",
        "payload_bits": np.array([], dtype=np.uint8),
        "protocol_version": None,
        "frame_type": None,
        "frame_type_name": None,
        "sequence": None,
        "total_data_frames": None,
        "payload_len_bits": None,
        "message_len_bytes": None,
        "message_crc16": None,
        "packet_len_bits": None,
        "preamble_errors": None,
        "rx_crc": None,
        "calc_crc": None,
        "structure_ok": False,
        "error": None,
        "score": 10**9,
    }

    if bits.size < header_len + FRAME_CRC_BITS:
        result["error"] = "No hay suficientes bits para header Fase C + CRC."
        return result

    preamble_errors = int(np.count_nonzero(bits[:32] != PREAMBLE_BITS))

    protocol_version = _bits_to_int(bits[32:40])
    frame_type = _bits_to_int(bits[40:48])
    sequence = _bits_to_int(bits[48:64])
    total_data_frames = _bits_to_int(bits[64:80])
    payload_len = _bits_to_int(bits[80:96])
    message_len_bytes = _bits_to_int(bits[96:112])
    message_crc16 = _bits_to_int(bits[112:128])

    max_payload_len = bits.size - header_len - FRAME_CRC_BITS
    packet_len = header_len + payload_len + FRAME_CRC_BITS

    frame_type_names = dict(_safe_tx_constant("FRAME_TYPE_NAMES", FRAME_TYPE_NAMES))
    frame_type_name = frame_type_names.get(frame_type, f"UNKNOWN_{frame_type}")

    result.update({
        "protocol_version": protocol_version,
        "frame_type": frame_type,
        "frame_type_name": frame_type_name,
        "sequence": sequence,
        "total_data_frames": total_data_frames,
        "payload_len_bits": payload_len,
        "message_len_bytes": message_len_bytes,
        "message_crc16": message_crc16,
        "packet_len_bits": packet_len,
        "preamble_errors": preamble_errors,
    })

    structure_penalty = 0

    if protocol_version != expected_protocol_version:
        structure_penalty += 400

    if frame_type not in frame_type_names:
        structure_penalty += 400

    if total_data_frames <= 0:
        structure_penalty += 300

    if payload_len < 0 or payload_len > max_payload_len:
        result["error"] = (
            f"Longitud de payload invalida: {payload_len} bits. "
            f"Maximo posible: {max_payload_len} bits."
        )
        result["score"] = 5000 + preamble_errors * 50 + structure_penalty
        return result

    payload_start = header_len
    payload_end = payload_start + payload_len
    crc_start = payload_end
    crc_end = crc_start + FRAME_CRC_BITS

    protected_bits = bits[:payload_end]
    payload_bits = bits[payload_start:payload_end]
    rx_crc = _bits_to_int(bits[crc_start:crc_end])
    calc_crc = crc16_ccitt(_bits_to_bytes_rx(protected_bits))

    crc_ok = (
        rx_crc == calc_crc
        and preamble_errors == 0
        and protocol_version == expected_protocol_version
        and frame_type in frame_type_names
    )

    # Texto parcial: solo se decodifica en DATA y solo en bytes completos.
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
        "error": None if crc_ok else "CRC de frame invalido, preambulo invalido o header invalido.",
        "score": preamble_errors * 50 + structure_penalty + (0 if crc_ok else 800),
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
        print(f"OK CRC frame: {decoded.get('crc_ok')}")
        print(f"Tipo frame: {decoded.get('frame_type_name')} ({decoded.get('frame_type')})")
        print(f"Secuencia: {decoded.get('sequence')}")
        print(f"Total DATA frames: {decoded.get('total_data_frames')}")
        print(f"Payload bits: {decoded.get('payload_len_bits')}")
        print(f"Bytes mensaje total: {decoded.get('message_len_bytes')}")
        msg_crc = decoded.get('message_crc16')
        print(f"CRC16 mensaje esperado: 0x{msg_crc:04X}" if msg_crc is not None else "CRC16 mensaje esperado: None")
        print(f"Convencion Manchester: {decoded.get('convention')}")
        print(f"Pares Manchester invalidos: {decoded.get('invalid_manchester_pairs')}")
        print(f"Nivel bajo estimado: {sampled.get('low_level')}")
        print(f"Nivel alto estimado: {sampled.get('high_level')}")
        print(f"Errores de preambulo: {decoded.get('preamble_errors')}")
        print(f"CRC frame RX: {decoded.get('rx_crc')}")
        print(f"CRC frame calculado: {decoded.get('calc_crc')}")
        print(f"Confianza estimada: {_estimate_frame_confidence(decoded):.3f}")
        print(f"Correccion erasures usada: {decoded.get('erasure_correction_used')}")
        print(f"Erasures corregidos: {decoded.get('corrected_erasures')}")
        print("Texto parcial del payload:")
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



# ============================================================
# 11. Receptor continuo Fase C
# ============================================================

@dataclass
class RxFrameObservation:
    """Lectura valida de un frame Fase C."""
    frame_type: int
    frame_type_name: str
    sequence: int
    total_data_frames: int
    payload_len_bits: int
    message_len_bytes: int
    message_crc16: int
    payload_bits: np.ndarray
    confidence: float
    timestamp_s: float
    capture_index: int
    decoded: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RxCollectorStatus:
    state: str = "WAITING_SYNC"
    sync_seen: int = 0
    end_seen: int = 0
    valid_packets_seen: int = 0
    duplicate_packets_seen: int = 0
    bad_packets_seen: int = 0
    started_at_s: Optional[float] = None
    last_valid_type: str = ""
    last_valid_sequence: Optional[int] = None
    last_error: str = ""


class RxFrameCollector:
    """
    Acumulador de frames Fase C.

    Reglas:
        - Acepta solo frames con CRC de frame valido.
        - Usa SYNC para pasar de WAITING_SYNC a RECEIVING.
        - Guarda DATA por numero de secuencia.
        - Si llega un duplicado de la misma secuencia, conserva el de mayor confianza.
        - Finaliza al tener todos los DATA y, si require_end=True, al observar END.
        - Verifica CRC global del mensaje antes de entregar texto.
    """
    def __init__(
        self,
        require_sync: bool = True,
        require_end: bool = True,
        encoding: str = "utf-8",
    ) -> None:
        self.require_sync = bool(require_sync)
        self.require_end = bool(require_end)
        self.encoding = encoding
        self.status = RxCollectorStatus()

        if not self.require_sync:
            self.status.state = "RECEIVING"

        self.total_data_frames: Optional[int] = None
        self.message_len_bytes: Optional[int] = None
        self.message_crc16: Optional[int] = None
        self.data_frames: Dict[int, RxFrameObservation] = {}
        self.sync_frames: Dict[int, RxFrameObservation] = {}
        self.end_frames: Dict[int, RxFrameObservation] = {}
        self.final_result: Optional[Dict[str, Any]] = None

    def reset(self) -> None:
        require_sync = self.require_sync
        require_end = self.require_end
        encoding = self.encoding
        self.__init__(require_sync=require_sync, require_end=require_end, encoding=encoding)

    def _metadata_matches(self, obs: RxFrameObservation) -> bool:
        if self.total_data_frames is None:
            return True
        return (
            obs.total_data_frames == self.total_data_frames
            and obs.message_len_bytes == self.message_len_bytes
            and obs.message_crc16 == self.message_crc16
        )

    def _adopt_metadata(self, obs: RxFrameObservation) -> None:
        if self.total_data_frames is None:
            self.total_data_frames = int(obs.total_data_frames)
            self.message_len_bytes = int(obs.message_len_bytes)
            self.message_crc16 = int(obs.message_crc16)

    def add_decoded(self, decoded: Dict[str, Any], timestamp_s: float, capture_index: int) -> Dict[str, Any]:
        """Agrega una decodificacion ya validada por CRC de frame."""
        if not decoded.get("crc_ok", False):
            self.status.bad_packets_seen += 1
            self.status.last_error = decoded.get("error") or "CRC frame invalido"
            return {"accepted": False, "reason": "crc_fail", "done": False}

        frame_type = int(decoded.get("frame_type"))
        sequence = int(decoded.get("sequence"))
        total_data_frames = int(decoded.get("total_data_frames"))
        payload_len_bits = int(decoded.get("payload_len_bits"))
        message_len_bytes = int(decoded.get("message_len_bytes"))
        message_crc16 = int(decoded.get("message_crc16"))
        payload_bits = np.asarray(decoded.get("payload_bits", []), dtype=np.uint8).copy()

        obs = RxFrameObservation(
            frame_type=frame_type,
            frame_type_name=str(decoded.get("frame_type_name", FRAME_TYPE_NAMES.get(frame_type, frame_type))),
            sequence=sequence,
            total_data_frames=total_data_frames,
            payload_len_bits=payload_len_bits,
            message_len_bytes=message_len_bytes,
            message_crc16=message_crc16,
            payload_bits=payload_bits,
            confidence=_estimate_frame_confidence(decoded),
            timestamp_s=timestamp_s,
            capture_index=capture_index,
            decoded=decoded,
        )

        self.status.valid_packets_seen += 1
        self.status.last_valid_type = obs.frame_type_name
        self.status.last_valid_sequence = obs.sequence

        if self.status.started_at_s is None:
            self.status.started_at_s = timestamp_s

        if obs.frame_type == FRAME_TYPE_SYNC:
            self._adopt_metadata(obs)
            self.status.sync_seen += 1
            self.sync_frames[obs.sequence] = obs
            if self.status.state == "WAITING_SYNC":
                self.status.state = "RECEIVING"
            return {"accepted": True, "reason": "sync", "done": False}

        if self.require_sync and self.status.state == "WAITING_SYNC":
            self.status.last_error = "DATA/END recibido antes de SYNC"
            return {"accepted": False, "reason": "waiting_sync", "done": False}

        if not self._metadata_matches(obs):
            self.status.last_error = "Header no coincide con la sesion actual"
            return {"accepted": False, "reason": "metadata_mismatch", "done": False}

        self._adopt_metadata(obs)

        if obs.frame_type == FRAME_TYPE_DATA:
            if obs.sequence < 0 or obs.sequence >= obs.total_data_frames:
                self.status.last_error = f"Secuencia DATA fuera de rango: {obs.sequence}"
                return {"accepted": False, "reason": "sequence_out_of_range", "done": False}

            previous = self.data_frames.get(obs.sequence)
            if previous is None:
                self.data_frames[obs.sequence] = obs
                action = "new_data"
            elif obs.confidence > previous.confidence:
                self.data_frames[obs.sequence] = obs
                self.status.duplicate_packets_seen += 1
                action = "duplicate_replaced"
            else:
                self.status.duplicate_packets_seen += 1
                action = "duplicate_ignored"

            done = self.try_finalize() is not None
            return {"accepted": True, "reason": action, "done": done}

        if obs.frame_type == FRAME_TYPE_END:
            self.status.end_seen += 1
            self.end_frames[obs.sequence] = obs
            done = self.try_finalize() is not None
            return {"accepted": True, "reason": "end", "done": done}

        self.status.last_error = f"Tipo de frame desconocido: {obs.frame_type}"
        return {"accepted": False, "reason": "unknown_type", "done": False}

    def missing_sequences(self) -> List[int]:
        if self.total_data_frames is None:
            return []
        return [i for i in range(self.total_data_frames) if i not in self.data_frames]

    def progress(self) -> Tuple[int, Optional[int]]:
        return len(self.data_frames), self.total_data_frames

    def try_finalize(self) -> Optional[Dict[str, Any]]:
        """Intenta reconstruir el mensaje; retorna resultado si ya esta completo."""
        if self.final_result is not None:
            return self.final_result

        if self.total_data_frames is None:
            return None

        if self.require_end and self.status.end_seen == 0:
            return None

        missing = self.missing_sequences()
        if missing:
            return None

        payload_bits = np.concatenate([
            self.data_frames[i].payload_bits for i in range(self.total_data_frames)
        ]).astype(np.uint8)

        payload_bytes_all = _bits_to_bytes_rx(payload_bits)
        message_len = int(self.message_len_bytes or 0)
        message_bytes = payload_bytes_all[:message_len]

        crc16_ccitt = _tx("crc16_ccitt")
        calc_message_crc = int(crc16_ccitt(message_bytes))
        expected_crc = int(self.message_crc16 or 0)
        crc_ok = (calc_message_crc == expected_crc)

        try:
            text = message_bytes.decode(self.encoding, errors="replace")
        except Exception:
            text = ""

        self.final_result = {
            "ok": bool(crc_ok),
            "text": text if crc_ok else "",
            "text_best_effort": text,
            "message_bytes": message_bytes,
            "message_len_bytes": message_len,
            "message_crc16_expected": expected_crc,
            "message_crc16_calc": calc_message_crc,
            "message_crc_ok": bool(crc_ok),
            "total_data_frames": self.total_data_frames,
            "received_data_frames": len(self.data_frames),
            "missing_sequences": [],
            "sync_seen": self.status.sync_seen,
            "end_seen": self.status.end_seen,
            "status": self.status,
            "data_frames": self.data_frames,
        }

        if crc_ok:
            self.status.state = "DONE"
        else:
            self.status.state = "CRC_GLOBAL_FAIL"

        return self.final_result

    def snapshot(self) -> Dict[str, Any]:
        received, total = self.progress()
        return {
            "state": self.status.state,
            "sync_seen": self.status.sync_seen,
            "end_seen": self.status.end_seen,
            "valid_packets_seen": self.status.valid_packets_seen,
            "duplicate_packets_seen": self.status.duplicate_packets_seen,
            "bad_packets_seen": self.status.bad_packets_seen,
            "received_data_frames": received,
            "total_data_frames": total,
            "missing_sequences": self.missing_sequences(),
            "last_valid_type": self.status.last_valid_type,
            "last_valid_sequence": self.status.last_valid_sequence,
            "last_error": self.status.last_error,
        }


def _set_camera_properties(cap: cv2.VideoCapture, width: int, height: int, disable_auto: bool = True) -> None:
    """Configura parametros basicos de camara cuando el driver lo permite."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    if disable_auto:
        # Estas propiedades no funcionan en todas las camaras/OS, pero si fallan no detienen el receptor.
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        except Exception:
            pass
        try:
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        except Exception:
            pass


def _draw_rx_hud(
    frame: np.ndarray,
    collector: RxFrameCollector,
    result: Optional[Dict[str, Any]],
    elapsed_s: float,
    fps_est: float,
    capture_index: int,
) -> np.ndarray:
    """Dibuja informacion de estado sobre el frame de preview."""
    if result is not None and result.get("overlay") is not None:
        out = result["overlay"].copy()
    else:
        out = frame.copy()

    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    snap = collector.snapshot()
    received = snap["received_data_frames"]
    total = snap["total_data_frames"]
    total_str = "?" if total is None else str(total)

    if result is not None and result.get("decoded") is not None:
        decoded = result["decoded"]
        last = f"last={decoded.get('frame_type_name')} seq={decoded.get('sequence')} crc={decoded.get('crc_ok')}"
    else:
        last = "last=None"

    lines = [
        "RX Fase C - q/ESC salir | r reset",
        f"state={snap['state']}  DATA={received}/{total_str}  SYNC={snap['sync_seen']}  END={snap['end_seen']}",
        f"valid={snap['valid_packets_seen']} dup={snap['duplicate_packets_seen']} bad={snap['bad_packets_seen']}  t={elapsed_s:.1f}s fps={fps_est:.1f}",
        last,
    ]

    y = 28
    for line in lines:
        cv2.putText(out, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        y += 26

    missing = snap["missing_sequences"]
    if missing:
        preview = ",".join(map(str, missing[:12]))
        if len(missing) > 12:
            preview += ",..."
        line = f"faltan: {preview}"
        cv2.putText(out, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)

    return out


def rx_from_camera_continuous(
    cfg: Any,
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    rectified_cell_size: Optional[int] = None,
    max_seconds: Optional[float] = 15.0,
    require_sync: bool = True,
    require_end: bool = True,
    process_every_n: int = 1,
    final_erasures: int = 0,
    max_marker_candidates: int = 8,
    show_preview: bool = True,
    preview_scale: float = 1.0,
    warmup_frames: int = 10,
    disable_camera_auto: bool = True,
    debug: bool = False,
    encoding: str = "utf-8",
    window_name: str = "RX Fase C - captura continua",
) -> Dict[str, Any]:
    """
    Captura video continuamente y reconstruye un mensaje Fase C.

    Parametros clave:
        require_sync:
            Si True, ignora DATA/END hasta observar al menos un SYNC valido.
        require_end:
            Si True, solo finaliza cuando ya recibio todos los DATA y un END valido.
        process_every_n:
            Procesa 1 de cada N capturas. N=1 es lo mas robusto, N=2 reduce carga CPU.
        final_erasures:
            Para recepcion continua conviene 0 inicialmente. Como se capturan varias copias
            de cada frame, es mas rapido esperar una lectura limpia que probar demasiadas
            combinaciones de erasures.

    Retorna un dict con:
        ok, text, message_crc_ok, missing_sequences, data_frames, stats, last_result.
    """
    if process_every_n < 1:
        raise ValueError("process_every_n debe ser >= 1.")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la camara con indice {camera_index}. "
            "Prueba camera_index=1 o revisa permisos de camara."
        )

    _set_camera_properties(cap, width, height, disable_auto=disable_camera_auto)

    collector = RxFrameCollector(require_sync=require_sync, require_end=require_end, encoding=encoding)
    last_result: Optional[Dict[str, Any]] = None
    start = time.perf_counter()
    last_fps_t = start
    fps_est = 0.0
    capture_index = 0
    processed_count = 0
    valid_add_events: List[Dict[str, Any]] = []

    try:
        for _ in range(warmup_frames):
            cap.read()
            time.sleep(0.01)

        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("La camara no entrego frame.")

            capture_index += 1
            now = time.perf_counter()
            elapsed = now - start

            if max_seconds is not None and elapsed >= max_seconds:
                break

            if capture_index % process_every_n == 0:
                processed_count += 1
                try:
                    last_result = process_captured_frame(
                        frame,
                        cfg,
                        rectified_cell_size=rectified_cell_size,
                        max_marker_candidates=max_marker_candidates,
                        final_erasures=final_erasures,
                        debug=False,
                        encoding=encoding,
                    )

                    decoded = last_result.get("decoded")
                    if decoded is not None and decoded.get("crc_ok", False):
                        add_info = collector.add_decoded(decoded, timestamp_s=elapsed, capture_index=capture_index)
                        valid_add_events.append(add_info)

                        if debug:
                            print(
                                f"[{elapsed:6.2f}s] {decoded.get('frame_type_name')} "
                                f"seq={decoded.get('sequence')} crc_ok "
                                f"conf={_estimate_frame_confidence(decoded):.3f} -> {add_info['reason']}"
                            )

                        if add_info.get("done"):
                            break

                except Exception as exc:
                    # En tiempo real no conviene detener por un frame malo.
                    collector.status.bad_packets_seen += 1
                    collector.status.last_error = str(exc)
                    if debug:
                        print(f"Frame {capture_index}: error RX: {exc}")

            if show_preview:
                dt = now - last_fps_t
                if dt > 0:
                    fps_est = 0.9 * fps_est + 0.1 * (1.0 / dt) if fps_est > 0 else 1.0 / dt
                last_fps_t = now

                preview = _draw_rx_hud(frame, collector, last_result, elapsed, fps_est, capture_index)
                if preview_scale != 1.0:
                    preview = cv2.resize(
                        preview,
                        None,
                        fx=float(preview_scale),
                        fy=float(preview_scale),
                        interpolation=cv2.INTER_AREA,
                    )
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF

                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    collector.reset()
                    valid_add_events.clear()
                    start = time.perf_counter()
                    last_fps_t = start
                    capture_index = 0
                    processed_count = 0
                    last_result = None

        final = collector.try_finalize()
        elapsed_total = time.perf_counter() - start

        if final is None:
            snap = collector.snapshot()
            final = {
                "ok": False,
                "text": "",
                "text_best_effort": "",
                "message_crc_ok": False,
                "message_crc16_expected": collector.message_crc16,
                "message_crc16_calc": None,
                "total_data_frames": collector.total_data_frames,
                "received_data_frames": len(collector.data_frames),
                "missing_sequences": collector.missing_sequences(),
                "sync_seen": collector.status.sync_seen,
                "end_seen": collector.status.end_seen,
                "status": collector.status,
                "data_frames": collector.data_frames,
                "snapshot": snap,
            }

        final.update({
            "elapsed_s": elapsed_total,
            "capture_frames_seen": capture_index,
            "processed_frames": processed_count,
            "valid_add_events": valid_add_events,
            "last_result": last_result,
            "collector": collector,
        })

        if final.get("ok"):
            print("Recepcion Fase C exitosa.")
            print(f"DATA recibidos: {final.get('received_data_frames')}/{final.get('total_data_frames')}")
            print(f"Tiempo RX: {elapsed_total:.2f} s")
        else:
            print("Recepcion Fase C incompleta o CRC global fallido.")
            print(f"DATA recibidos: {final.get('received_data_frames')}/{final.get('total_data_frames')}")
            print(f"Faltantes: {final.get('missing_sequences')}")
            print(f"SYNC vistos: {final.get('sync_seen')} | END vistos: {final.get('end_seen')}")

        return final

    finally:
        cap.release()
        if show_preview:
            cv2.destroyWindow(window_name)
            for _ in range(3):
                cv2.waitKey(1)


def rx_from_video_file(
    path: str,
    cfg: Any,
    rectified_cell_size: Optional[int] = None,
    require_sync: bool = True,
    require_end: bool = True,
    process_every_n: int = 1,
    final_erasures: int = 0,
    max_seconds: Optional[float] = None,
    debug: bool = False,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Variante util para probar el receptor usando un video guardado por save_frames_as_video(...).
    No abre preview; procesa frames del archivo hasta finalizar o terminar el video.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {path}")

    collector = RxFrameCollector(require_sync=require_sync, require_end=require_end, encoding=encoding)
    start = time.perf_counter()
    capture_index = 0
    processed_count = 0
    last_result = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            capture_index += 1
            elapsed = time.perf_counter() - start
            if max_seconds is not None and elapsed >= max_seconds:
                break
            if capture_index % process_every_n != 0:
                continue

            processed_count += 1
            last_result = process_captured_frame(
                frame,
                cfg,
                rectified_cell_size=rectified_cell_size,
                max_marker_candidates=8,
                final_erasures=final_erasures,
                debug=False,
                encoding=encoding,
            )
            decoded = last_result.get("decoded")
            if decoded is not None and decoded.get("crc_ok", False):
                add_info = collector.add_decoded(decoded, timestamp_s=elapsed, capture_index=capture_index)
                if debug:
                    print(
                        f"frame_video={capture_index} {decoded.get('frame_type_name')} "
                        f"seq={decoded.get('sequence')} -> {add_info['reason']}"
                    )
                if add_info.get("done"):
                    break

        final = collector.try_finalize()
        if final is None:
            final = {
                "ok": False,
                "text": "",
                "text_best_effort": "",
                "message_crc_ok": False,
                "missing_sequences": collector.missing_sequences(),
                "received_data_frames": len(collector.data_frames),
                "total_data_frames": collector.total_data_frames,
                "sync_seen": collector.status.sync_seen,
                "end_seen": collector.status.end_seen,
            }
        final.update({
            "elapsed_s": time.perf_counter() - start,
            "capture_frames_seen": capture_index,
            "processed_frames": processed_count,
            "last_result": last_result,
            "collector": collector,
        })
        return final
    finally:
        cap.release()

__all__ = [
    "bind_tx_helpers",
    "bind_tx_module",
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
    "rx_from_camera_continuous",
    "rx_from_video_file",
    "RxFrameCollector",
    "RxFrameObservation",
    "RxCollectorStatus",
    "MarkerCandidate",
]
