"""
rx_video_module_fase_c_4ask.py

Receptor Fase C para el módem óptico pantalla-cámara usando únicamente
modulación 4-ASK en escala de grises.

Restricción de este módulo:
- NO captura cámara en vivo.
- NO hace captura continua + decodificación simultánea.
- La entrada principal es un archivo de video ya grabado, mediante:

      rx_from_video_file(...)

Cadena de procesamiento por frame de video:
    detección de marcadores fiduciales
    estimación de homografía
    rectificación de grilla
    calibración de brillo con pilotos PILOT_0..PILOT_3
    muestreo de celdas DATA
    demapeo 4-ASK Gray-coded
    validación CRC por frame
    acumulación SYNC/DATA/END
    reconstrucción y CRC global del mensaje

Uso típico en Jupyter:

    import importlib
    import modulo_1_fase_c_tx_4ask as tx4
    import rx_video_module_fase_c_4ask as rx4

    importlib.reload(tx4)
    importlib.reload(rx4)

    cfg = tx4.TxVisualConfig(frame_duration_s=0.12)
    rx4.bind_tx_module(tx4)

    result = rx4.rx_from_video_file(
        "outputs/fase_c_4ask/tx_sequence_4ask.mp4",
        cfg,
        require_sync=True,
        require_end=True,
        process_every_n=1,
        debug=True,
    )

    print(result["text"])
"""

from __future__ import annotations

import cv2
import numpy as np
import itertools
import time
import sys
import importlib

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional, Sequence


# ============================================================
# 0. Enlace con funciones/constantes del transmisor 4-ASK
# ============================================================

_TX_BINDINGS: Dict[str, Any] = {}
_TX_MODULE: Optional[Any] = None


def bind_tx_helpers(**kwargs: Any) -> None:
    """
    Registra manualmente funciones/constantes del transmisor 4-ASK.

    Normalmente no hace falta usar esta función si se llama:
        rx4.bind_tx_module(tx4)
    """
    _TX_BINDINGS.update(kwargs)


def bind_tx_module(tx_module: Any) -> None:
    """
    Registra el módulo transmisor 4-ASK completo.

    Uso recomendado:
        import modulo_1_fase_c_tx_4ask as tx4
        import rx_video_module_fase_c_4ask as rx4
        rx4.bind_tx_module(tx4)
    """
    global _TX_MODULE
    _TX_MODULE = tx_module


def _tx(name: str) -> Any:
    """
    Busca una función/constante del transmisor 4-ASK.
    Orden:
        1) bindings manuales
        2) módulo registrado con bind_tx_module(...)
        3) namespace principal del notebook
        4) import automático de modulo_1_fase_c_tx_4ask
    """
    if name in _TX_BINDINGS:
        return _TX_BINDINGS[name]

    if _TX_MODULE is not None and hasattr(_TX_MODULE, name):
        return getattr(_TX_MODULE, name)

    main = sys.modules.get("__main__")
    if main is not None and hasattr(main, name):
        return getattr(main, name)

    try:
        tx_mod = importlib.import_module("modulo_1_fase_c_tx_4ask")
        if hasattr(tx_mod, name):
            return getattr(tx_mod, name)
    except Exception:
        pass

    raise NameError(
        f"No se encontró '{name}'. Importa el transmisor 4-ASK y registra sus "
        f"dependencias con rx4.bind_tx_module(tx4) o rx4.bind_tx_helpers(...)."
    )


def _safe_tx_constant(name: str, fallback: Any) -> Any:
    """Retorna una constante del TX si existe; si no, usa fallback."""
    try:
        return _tx(name)
    except NameError:
        return fallback


# ============================================================
# 0.1 Constantes locales del protocolo Fase C 4-ASK
# ============================================================

FRAME_TYPE_SYNC = 1
FRAME_TYPE_DATA = 2
FRAME_TYPE_END = 3

FRAME_TYPE_NAMES = {
    FRAME_TYPE_SYNC: "SYNC",
    FRAME_TYPE_DATA: "DATA",
    FRAME_TYPE_END: "END",
}

# En el TX 4-ASK se usa versión 2 para evitar confusión con BPSK/Manchester.
PROTOCOL_VERSION = 2
HEADER_BITS_NO_PAYLOAD = 32 + 8 + 8 + 16 + 16 + 16 + 16 + 16
FRAME_CRC_BITS = 16
BITS_PER_SYMBOL = 2
GRAY_LEVELS = [50, 110, 170, 230]

# Mapeo Gray inverso usado por el transmisor:
# símbolo 0 -> 00, símbolo 1 -> 01, símbolo 2 -> 11, símbolo 3 -> 10.
SYMBOL_TO_BITS_4ASK_FALLBACK = {
    0: (0, 0),
    1: (0, 1),
    2: (1, 1),
    3: (1, 0),
}

PILOT_ROLE_TO_LEVEL_INDEX_FALLBACK = {
    "PILOT_0": 0,
    "PILOT_1": 1,
    "PILOT_2": 2,
    "PILOT_3": 3,
}


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

ANCHOR_PATTERN_A = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1],
], dtype=np.uint8)

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
    """Convierte bits a bytes, usando el helper del TX 4-ASK si existe."""
    bits = np.asarray(bits, dtype=np.uint8).flatten()

    try:
        bits_to_bytes = _tx("bits_to_bytes")
        return bytes(bits_to_bytes(bits))
    except NameError:
        pass

    pad = (-bits.size) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

    return np.packbits(bits).tobytes()


def _ideal_marker_centers(cfg: Any, rectified_cell_size: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """Centros ideales de los marcadores dentro de la grilla rectificada."""
    marker_origins = _tx("marker_origins")
    origins = list(marker_origins(cfg))

    centers = []
    for r0, c0 in origins:
        x = (c0 + 3.5) * rectified_cell_size
        y = (r0 + 3.5) * rectified_cell_size
        centers.append([x, y])

    return np.asarray(centers, dtype=np.float32), origins


# ============================================================
# 3. Detección de marcadores fiduciales
# ============================================================

def _sample_7x7_pattern(marker_gray: np.ndarray, margin_frac: float = 0.25) -> np.ndarray:
    """Toma un marcador rectificado y estima su patrón binario 7x7."""
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
    Compara un patrón 7x7 contra los patrones conocidos.
    Retorna:
        marker_type, error_hamming, rotación_k
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
    """Detecta candidatos a marcadores fiduciales en una imagen de video."""
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

        candidates.append(MarkerCandidate(
            center=center,
            quad=quad,
            area=area,
            marker_type=marker_type,
            pattern_error=err,
            rotation_k=rot_k,
            sampled_pattern=sampled,
        ))

    candidates.sort(
        key=lambda c: (
            c.pattern_error,
            0 if c.marker_type == "anchor" else 1,
            -c.area,
        )
    )

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
# 4. Calibración de brillo, muestreo y demapeo 4-ASK
# ============================================================

def _sample_cell_center(
    gray: np.ndarray,
    row: int,
    col: int,
    cell_px: int,
    margin_frac: float = 0.30,
) -> float:
    """
    Muestra una celda usando solo su región central.
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


def _fallback_centers_from_data(data_values: np.ndarray) -> np.ndarray:
    """Centros 4-ASK aproximados cuando no hay suficientes pilotos válidos."""
    valid = data_values[np.isfinite(data_values)]
    if valid.size == 0:
        return np.asarray(GRAY_LEVELS, dtype=np.float32)
    if valid.size < 8:
        lo = float(np.nanmin(valid))
        hi = float(np.nanmax(valid))
        if abs(hi - lo) < 1.0:
            hi = lo + 3.0
        return np.linspace(lo, hi, 4, dtype=np.float32)
    return np.percentile(valid, [8, 36, 64, 92]).astype(np.float32)


def _make_strictly_increasing(values: Sequence[float], min_sep: float = 1.0) -> np.ndarray:
    """Ordena y fuerza separación mínima para evitar umbrales degenerados."""
    centers = np.asarray(values, dtype=np.float32).copy()
    centers = np.sort(centers)
    for i in range(1, centers.size):
        if centers[i] <= centers[i - 1] + min_sep:
            centers[i] = centers[i - 1] + min_sep
    return centers


def calibrate_and_sample_symbols_4ask(
    warped_grid: np.ndarray,
    cfg: Any,
    rectified_cell_size: Optional[int] = None,
    margin_frac: float = 0.30,
) -> Dict[str, Any]:
    """
    Calibra brillo usando pilotos de cuatro niveles y muestrea DATA.

    Retorna símbolos 4-ASK 0..3, valores muestreados, centros de decisión,
    umbrales adaptativos y una confianza por símbolo.
    """
    cell_px = int(rectified_cell_size or cfg.cell_size)
    gray = _as_gray(warped_grid)

    build_role_grid = _tx("build_role_grid")
    data_positions = _tx("data_positions")
    pilot_map = dict(_safe_tx_constant("PILOT_ROLE_TO_LEVEL_INDEX", PILOT_ROLE_TO_LEVEL_INDEX_FALLBACK))

    roles = build_role_grid(cfg)
    positions = data_positions(cfg)

    pilot_values: Dict[int, List[float]] = {0: [], 1: [], 2: [], 3: []}

    for r in range(cfg.grid_rows):
        for c in range(cfg.grid_cols):
            role = roles[r, c]
            if role not in pilot_map:
                continue
            val = _sample_cell_center(gray, r, c, cell_px, margin_frac)
            if np.isfinite(val):
                idx = int(pilot_map[role])
                if idx in pilot_values:
                    pilot_values[idx].append(val)

    data_values = []
    for r, c in positions:
        data_values.append(_sample_cell_center(gray, r, c, cell_px, margin_frac))
    data_values = np.asarray(data_values, dtype=np.float32)

    raw_centers = np.full(4, np.nan, dtype=np.float32)
    pilot_counts = np.zeros(4, dtype=np.int32)
    for idx in range(4):
        vals = np.asarray(pilot_values[idx], dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        pilot_counts[idx] = vals.size
        if vals.size:
            raw_centers[idx] = float(np.median(vals))

    if np.all(np.isfinite(raw_centers)):
        centers = raw_centers.astype(np.float32)
    else:
        fallback = _fallback_centers_from_data(data_values)
        centers = raw_centers.copy()
        missing = ~np.isfinite(centers)
        centers[missing] = fallback[missing]

    # Los pilotos están asociados a niveles crecientes; si el canal/ruido causa
    # inversiones, se usa el orden de brillo para recuperar umbrales robustos.
    centers = _make_strictly_increasing(centers, min_sep=1.0)
    thresholds = 0.5 * (centers[:-1] + centers[1:])

    # Decisión por umbrales adaptativos: 0,1,2,3.
    symbols = np.digitize(data_values, thresholds).astype(np.uint8)
    symbols = np.clip(symbols, 0, 3).astype(np.uint8)

    # Confianza basada en la distancia al umbral más cercano.
    finite_thresholds = thresholds[np.isfinite(thresholds)]
    if finite_thresholds.size:
        distance_to_boundary = np.min(np.abs(data_values[:, None] - finite_thresholds[None, :]), axis=1)
        local_scale = max(1.0, 0.5 * float(np.min(np.diff(centers))))
        confidence = np.clip(distance_to_boundary / local_scale, 0.0, 1.0)
    else:
        confidence = np.zeros_like(data_values, dtype=np.float32)

    confidence[~np.isfinite(data_values)] = 0.0

    # Imagen calibrada aproximada, útil para depuración visual.
    lo = centers[0]
    hi = centers[-1]
    span = max(1.0, float(hi - lo))
    calibrated_gray = np.clip((gray.astype(np.float32) - lo) * 255.0 / span, 0, 255).astype(np.uint8)

    return {
        "symbols": symbols,
        "values": data_values,
        "confidence": confidence.astype(np.float32),
        "centers": centers.astype(np.float32),
        "thresholds": thresholds.astype(np.float32),
        "pilot_values": pilot_values,
        "pilot_counts": pilot_counts,
        "calibrated_gray": calibrated_gray,
        "data_positions": positions,
        "gray_levels_tx": tuple(_safe_tx_constant("GRAY_LEVELS", GRAY_LEVELS)),
    }


def symbols_4ask_to_bits(symbols: np.ndarray) -> np.ndarray:
    """Convierte símbolos 4-ASK 0..3 a bits usando Gray inverso."""
    symbols = np.asarray(symbols, dtype=np.uint8).flatten()

    if not np.all((symbols >= 0) & (symbols <= 3)):
        raise ValueError("Los símbolos 4-ASK deben estar en el rango 0..3.")

    # Usa el helper del TX si está disponible; así se evita cualquier desacople.
    try:
        fn = _tx("symbols_4ask_to_bits")
        return np.asarray(fn(symbols), dtype=np.uint8).flatten()
    except NameError:
        pass

    mapping = dict(_safe_tx_constant("SYMBOL_TO_BITS_4ASK", SYMBOL_TO_BITS_4ASK_FALLBACK))
    bits = np.empty(symbols.size * BITS_PER_SYMBOL, dtype=np.uint8)
    for i, symbol in enumerate(symbols):
        b0, b1 = mapping[int(symbol)]
        bits[2 * i] = int(b0)
        bits[2 * i + 1] = int(b1)
    return bits


# ============================================================
# 5. CRC, parsing del paquete y decodificación 4-ASK
# ============================================================

def _parse_packet_bits(bits: np.ndarray, encoding: str = "utf-8") -> Dict[str, Any]:
    """
    Interpreta el paquete Fase C 4-ASK:
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
    """
    bits = np.asarray(bits, dtype=np.uint8).flatten()

    PREAMBLE_BITS = np.asarray(_tx("PREAMBLE_BITS"), dtype=np.uint8).flatten()
    crc16_ccitt = _tx("crc16_ccitt")

    expected_protocol_version = int(_safe_tx_constant("PROTOCOL_VERSION", PROTOCOL_VERSION))
    header_len = int(_safe_tx_constant("HEADER_BITS_NO_PAYLOAD", HEADER_BITS_NO_PAYLOAD))
    frame_crc_bits = int(_safe_tx_constant("FRAME_CRC_BITS", FRAME_CRC_BITS))

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

    if bits.size < header_len + frame_crc_bits:
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

    max_payload_len = bits.size - header_len - frame_crc_bits
    packet_len = header_len + payload_len + frame_crc_bits

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
    if payload_len % 8 != 0:
        # El TX 4-ASK byte-alinea los payloads para simplificar reconstrucción.
        structure_penalty += 100

    if payload_len < 0 or payload_len > max_payload_len:
        result["error"] = (
            f"Longitud de payload inválida: {payload_len} bits. "
            f"Máximo posible: {max_payload_len} bits."
        )
        result["score"] = 5000 + preamble_errors * 50 + structure_penalty
        return result

    payload_start = header_len
    payload_end = payload_start + payload_len
    crc_start = payload_end
    crc_end = crc_start + frame_crc_bits

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
        "error": None if crc_ok else "CRC de frame inválido, preámbulo inválido o header inválido.",
        "score": preamble_errors * 50 + structure_penalty + (0 if crc_ok else 800),
    })

    return result


def decode_warped_grid_4ask(
    warped_grid: np.ndarray,
    cfg: Any,
    rectified_cell_size: Optional[int] = None,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """Decodifica una grilla ya rectificada usando únicamente 4-ASK."""
    sampled = calibrate_and_sample_symbols_4ask(
        warped_grid,
        cfg,
        rectified_cell_size=rectified_cell_size,
    )

    bits = symbols_4ask_to_bits(sampled["symbols"])
    parsed = _parse_packet_bits(bits, encoding=encoding)

    conf_arr = np.asarray(sampled.get("confidence", []), dtype=np.float32)
    confidence_penalty = int(np.nanmean(1.0 - conf_arr) * 100) if conf_arr.size else 100
    center_gaps = np.diff(sampled["centers"])
    min_gap = float(np.min(center_gaps)) if center_gaps.size else 0.0
    spacing_penalty = 0 if min_gap >= 10 else int((10 - min_gap) * 10)

    parsed.update({
        "score": parsed["score"] + confidence_penalty + spacing_penalty,
        "sampled": sampled,
        "modulation": "4-ASK grayscale Gray-coded",
        "bits_from_symbols": bits,
        "symbol_count": int(sampled["symbols"].size),
        "mean_symbol_confidence": float(np.nanmean(conf_arr)) if conf_arr.size else 0.0,
        "min_center_gap": min_gap,
    })

    return parsed


# Alias explícito para mantener una llamada parecida al receptor anterior,
# pero aquí NO hay Manchester/BPSK.
decode_warped_grid = decode_warped_grid_4ask


# ============================================================
# 6. Homografía, rectificación y búsqueda de orientación
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


def process_video_frame(
    frame: np.ndarray,
    cfg: Any,
    rectified_cell_size: Optional[int] = None,
    max_marker_candidates: int = 8,
    debug: bool = False,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Procesa un frame individual ya leído desde un archivo de video.

    Esta función no captura cámara. Es auxiliar de rx_from_video_file(...).
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

                decoded = decode_warped_grid_4ask(
                    warped,
                    cfg,
                    rectified_cell_size=cell_px,
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
            "error": "No se pudo construir una homografía válida.",
            "text": "",
            "captured": frame,
            "overlay": overlay,
            "markers": markers,
            "warped": None,
            "decoded": None,
        }

    evaluations.sort(key=lambda ev: ev["score"])
    best_ev = evaluations[0]
    best_warped = cv2.warpPerspective(frame, best_ev["H"], (dst_w, dst_h))
    best_decoded = best_ev["decoded"]

    return {
        "ok": False,
        "error": "No se logró validar CRC en este frame de video.",
        "text": best_decoded.get("text", "") if best_decoded else "",
        "captured": frame,
        "overlay": overlay,
        "markers": markers,
        "warped": best_warped,
        "decoded": best_decoded,
        "H": best_ev["H"],
        "combo": best_ev["combo"],
        "perm": best_ev["perm"],
    }


# ============================================================
# 7. Visualización de diagnóstico opcional
# ============================================================

def show_rx_result(result: Dict[str, Any]) -> None:
    """Muestra diagnóstico visual de un frame procesado desde video."""
    if plt is None:
        raise RuntimeError("matplotlib no está disponible en este entorno.")

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
    plt.title("Frame de video")
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
        centers = sampled.get("centers")
        thresholds = sampled.get("thresholds")
        pilot_counts = sampled.get("pilot_counts")
        print("========== Diagnóstico RX 4-ASK ==========")
        print(f"OK CRC frame: {decoded.get('crc_ok')}")
        print(f"Modulación: {decoded.get('modulation')}")
        print(f"Tipo frame: {decoded.get('frame_type_name')} ({decoded.get('frame_type')})")
        print(f"Secuencia: {decoded.get('sequence')}")
        print(f"Total DATA frames: {decoded.get('total_data_frames')}")
        print(f"Payload bits: {decoded.get('payload_len_bits')}")
        print(f"Bytes mensaje total: {decoded.get('message_len_bytes')}")
        msg_crc = decoded.get('message_crc16')
        print(f"CRC16 mensaje esperado: 0x{msg_crc:04X}" if msg_crc is not None else "CRC16 mensaje esperado: None")
        print(f"Centros 4-ASK estimados: {centers}")
        print(f"Umbrales 4-ASK estimados: {thresholds}")
        print(f"Pilotos por nivel: {pilot_counts}")
        print(f"Errores de preámbulo: {decoded.get('preamble_errors')}")
        print(f"CRC frame RX: {decoded.get('rx_crc')}")
        print(f"CRC frame calculado: {decoded.get('calc_crc')}")
        print(f"Confianza media símbolos: {decoded.get('mean_symbol_confidence'):.3f}")
        print("Texto parcial del payload:")
        print(decoded.get("text", ""))


# ============================================================
# 8. Acumulador de frames Fase C
# ============================================================

@dataclass
class RxFrameObservation:
    """Lectura válida de un frame Fase C."""
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
    Acumulador de frames Fase C 4-ASK.

    Reglas:
        - Acepta solo frames con CRC de frame válido.
        - Usa SYNC para pasar de WAITING_SYNC a RECEIVING.
        - Guarda DATA por número de secuencia.
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
        """Agrega una decodificación ya validada por CRC de frame."""
        if not decoded.get("crc_ok", False):
            self.status.bad_packets_seen += 1
            self.status.last_error = decoded.get("error") or "CRC frame inválido"
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
            self.status.last_error = "Header no coincide con la sesión actual"
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
        """Intenta reconstruir el mensaje; retorna resultado si ya está completo."""
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


def _estimate_frame_confidence(decoded: Dict[str, Any]) -> float:
    """Métrica simple de confianza para comparar lecturas duplicadas."""
    sampled = decoded.get("sampled", {}) if isinstance(decoded, dict) else {}
    conf_arr = sampled.get("confidence")

    if conf_arr is None:
        base = 0.0
    else:
        arr = np.asarray(conf_arr, dtype=np.float32)
        base = float(np.nanmean(arr)) if arr.size else 0.0

    preamble_errors = int(decoded.get("preamble_errors") or 0)
    min_gap = float(decoded.get("min_center_gap") or 0.0)
    gap_penalty = 0.0 if min_gap >= 10.0 else min(0.25, (10.0 - min_gap) / 40.0)

    penalty = 0.03 * preamble_errors + gap_penalty
    if not decoded.get("crc_ok", False):
        penalty += 0.5

    return float(np.clip(base - penalty, 0.0, 1.0))


# ============================================================
# 9. Función principal: decodificación desde video ya grabado
# ============================================================

def rx_from_video_file(
    path: str,
    cfg: Any,
    rectified_cell_size: Optional[int] = None,
    require_sync: bool = True,
    require_end: bool = True,
    process_every_n: int = 1,
    max_marker_candidates: int = 8,
    max_video_seconds: Optional[float] = None,
    stop_when_done: bool = True,
    debug: bool = False,
    show_last_result: bool = False,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Decodifica un mensaje Fase C 4-ASK desde un archivo de video ya capturado.

    Parámetros clave:
        path:
            Ruta del archivo de video grabado previamente.
        cfg:
            Configuración del transmisor 4-ASK usada para generar la grilla.
        require_sync:
            Si True, ignora DATA/END hasta observar al menos un SYNC válido.
        require_end:
            Si True, solo finaliza cuando recibió todos los DATA y un END válido.
        process_every_n:
            Procesa 1 de cada N frames del video. N=1 es lo más robusto.
        max_video_seconds:
            Límite opcional medido en tiempo interno del video, no en tiempo real.
        stop_when_done:
            Si True, se detiene tan pronto reconstruye y valida el mensaje.
        show_last_result:
            Si True, muestra diagnóstico visual del último frame procesado.

    Retorna un dict con:
        ok, text, message_crc_ok, missing_sequences, data_frames, stats, last_result.
    """
    if process_every_n < 1:
        raise ValueError("process_every_n debe ser >= 1.")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {path}")

    collector = RxFrameCollector(require_sync=require_sync, require_end=require_end, encoding=encoding)
    start_wall = time.perf_counter()
    capture_index = 0
    processed_count = 0
    last_result: Optional[Dict[str, Any]] = None
    valid_add_events: List[Dict[str, Any]] = []

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_duration_s = (total_video_frames / fps) if fps > 0 and total_video_frames > 0 else None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            capture_index += 1

            pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            video_time_s = pos_msec / 1000.0
            if video_time_s <= 0.0 and fps > 0:
                video_time_s = (capture_index - 1) / fps

            if max_video_seconds is not None and video_time_s >= max_video_seconds:
                break

            if capture_index % process_every_n != 0:
                continue

            processed_count += 1

            try:
                last_result = process_video_frame(
                    frame,
                    cfg,
                    rectified_cell_size=rectified_cell_size,
                    max_marker_candidates=max_marker_candidates,
                    debug=False,
                    encoding=encoding,
                )
            except Exception as exc:
                collector.status.bad_packets_seen += 1
                collector.status.last_error = str(exc)
                if debug:
                    print(f"frame_video={capture_index}: error RX: {exc}")
                continue

            decoded = last_result.get("decoded") if last_result is not None else None
            if decoded is not None and decoded.get("crc_ok", False):
                add_info = collector.add_decoded(decoded, timestamp_s=video_time_s, capture_index=capture_index)
                valid_add_events.append({
                    **add_info,
                    "capture_index": capture_index,
                    "video_time_s": video_time_s,
                    "frame_type_name": decoded.get("frame_type_name"),
                    "sequence": decoded.get("sequence"),
                    "confidence": _estimate_frame_confidence(decoded),
                })

                if debug:
                    received, total = collector.progress()
                    total_str = "?" if total is None else str(total)
                    print(
                        f"frame_video={capture_index:05d} t={video_time_s:7.3f}s "
                        f"{decoded.get('frame_type_name')} seq={decoded.get('sequence')} "
                        f"-> {add_info['reason']} DATA={received}/{total_str} "
                        f"conf={_estimate_frame_confidence(decoded):.3f}"
                    )

                if stop_when_done and add_info.get("done"):
                    break
            else:
                collector.status.bad_packets_seen += 1
                if decoded is not None:
                    collector.status.last_error = decoded.get("error") or "Frame sin CRC válido"

        final = collector.try_finalize()
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

        elapsed_wall_s = time.perf_counter() - start_wall

        final.update({
            "elapsed_wall_s": elapsed_wall_s,
            "capture_frames_seen": capture_index,
            "processed_frames": processed_count,
            "process_every_n": process_every_n,
            "video_fps_reported": fps,
            "video_frame_count_reported": total_video_frames,
            "video_duration_s_reported": video_duration_s,
            "valid_add_events": valid_add_events,
            "last_result": last_result,
            "collector": collector,
            "input_video_path": str(path),
            "modulation": "4-ASK grayscale Gray-coded",
        })

        if show_last_result and last_result is not None:
            show_rx_result(last_result)

        if debug:
            if final.get("ok"):
                print("Recepción Fase C 4-ASK desde video exitosa.")
                print(f"DATA recibidos: {final.get('received_data_frames')}/{final.get('total_data_frames')}")
                print(f"Frames de video vistos: {capture_index} | procesados: {processed_count}")
            else:
                print("Recepción Fase C 4-ASK desde video incompleta o CRC global fallido.")
                print(f"DATA recibidos: {final.get('received_data_frames')}/{final.get('total_data_frames')}")
                print(f"Faltantes: {final.get('missing_sequences')}")
                print(f"SYNC vistos: {final.get('sync_seen')} | END vistos: {final.get('end_seen')}")

        return final

    finally:
        cap.release()


__all__ = [
    "bind_tx_helpers",
    "bind_tx_module",
    "detect_finder_markers",
    "draw_marker_overlay",
    "calibrate_and_sample_symbols_4ask",
    "symbols_4ask_to_bits",
    "decode_warped_grid_4ask",
    "decode_warped_grid",
    "process_video_frame",
    "show_rx_result",
    "rx_from_video_file",
    "RxFrameCollector",
    "RxFrameObservation",
    "RxCollectorStatus",
    "MarkerCandidate",
]
