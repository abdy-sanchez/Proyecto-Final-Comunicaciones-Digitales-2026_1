"""
rx_video_module_fase_c_8csk.py

Receptor Fase C EXCLUSIVO para modulación 8-CSK y videos previamente
capturados. No abre la cámara y no contiene una ruta de recepción en vivo.

Cadena de recepción
-------------------
1. Lee cuadros desde un archivo de video con cv2.VideoCapture(path).
2. Detecta los cuatro marcadores fiduciales.
3. Estima la homografía y rectifica la grilla.
4. Muestra los pilotos RGB y estima una corrección afín de color:

       RGB_corregido = RGB_capturado @ A.T + b

5. Construye centroides 8-CSK por frame a partir de los pilotos corregidos.
6. Clasifica cada celda DATA por distancia euclidiana en cromaticidad rgb
   normalizada: r=R/(R+G+B), g=G/(R+G+B).
7. Convierte los índices 8-CSK a bits mediante las etiquetas Gray del TX.
8. Valida preámbulo, cabecera, CRC de frame y CRC global del mensaje.
9. Acumula SYNC | DATA... | END y reconstruye el texto.

Uso típico en Jupyter
---------------------
    import importlib
    import modulo_1_fase_c_tx_8csk as tx
    import rx_video_module_fase_c_8csk as rx

    importlib.reload(tx)
    importlib.reload(rx)

    cfg = tx.TxVisualConfig(frame_duration_s=0.12)
    rx.bind_tx_module(tx)

    result = rx.rx_from_video_file(
        "outputs/captura_rx.mp4",
        cfg,
        require_sync=True,
        require_end=True,
        process_every_n=1,
        reuse_homography=True,
        debug=True,
    )

    print(result["text"])

Convención de color
-------------------
OpenCV entrega imágenes en BGR. Dentro del calibrador y del clasificador se
trabaja explícitamente en RGB flotante.
"""

from __future__ import annotations

import importlib
import itertools
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

np.set_printoptions(precision=4, suppress=True)


# ============================================================
# 0. Enlace con el transmisor 8-CSK
# ============================================================

_TX_BINDINGS: Dict[str, Any] = {}
_TX_MODULE: Optional[Any] = None


def bind_tx_helpers(**kwargs: Any) -> None:
    """Registra manualmente funciones o constantes del transmisor."""
    _TX_BINDINGS.update(kwargs)


def bind_tx_module(tx_module: Any) -> None:
    """Registra el módulo transmisor 8-CSK completo."""
    global _TX_MODULE
    _TX_MODULE = tx_module


def _tx(name: str) -> Any:
    """Obtiene una función/constante del TX 8-CSK."""
    if name in _TX_BINDINGS:
        return _TX_BINDINGS[name]

    if _TX_MODULE is not None and hasattr(_TX_MODULE, name):
        return getattr(_TX_MODULE, name)

    main = sys.modules.get("__main__")
    if main is not None and hasattr(main, name):
        return getattr(main, name)

    for module_name in ("modulo_1_fase_c_tx_8csk",):
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, name):
                return getattr(module, name)
        except Exception:
            continue

    raise NameError(
        f"No se encontró '{name}' en el transmisor 8-CSK. "
        "Importa modulo_1_fase_c_tx_8csk como tx y ejecuta "
        "rx.bind_tx_module(tx)."
    )


def _safe_tx_constant(name: str, fallback: Any) -> Any:
    try:
        return _tx(name)
    except NameError:
        return fallback


# ============================================================
# 0.1 Fallbacks del protocolo y constelación
# ============================================================

FRAME_TYPE_SYNC = 1
FRAME_TYPE_DATA = 2
FRAME_TYPE_END = 3

FRAME_TYPE_NAMES = {
    FRAME_TYPE_SYNC: "SYNC",
    FRAME_TYPE_DATA: "DATA",
    FRAME_TYPE_END: "END",
}

PROTOCOL_VERSION = 2
HEADER_BITS_NO_PAYLOAD = 32 + 8 + 8 + 16 + 16 + 16 + 16 + 16
FRAME_CRC_BITS = 16
BITS_PER_SYMBOL = 3

PREAMBLE_BITS_FALLBACK = np.array(
    [int(b) for b in "10111000110100101110001001011010"],
    dtype=np.uint8,
)

GRAY_LABELS_FALLBACK: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 1),
    (0, 1, 0),
    (1, 1, 0),
    (1, 1, 1),
    (1, 0, 1),
    (1, 0, 0),
)


# ============================================================
# 1. Patrones y estructuras de datos
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

ANCHOR_PATTERN = np.array([
    [1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 1],
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


@dataclass
class ColorCalibration:
    """Modelo afín capturado RGB -> RGB de referencia."""

    matrix: np.ndarray              # shape (3, 3), se aplica x @ matrix.T
    bias: np.ndarray                # shape (3,)
    rmse_rgb: float
    inlier_mask: np.ndarray
    observed_pilots_rgb: np.ndarray
    expected_pilots_rgb: np.ndarray
    corrected_pilots_rgb: np.ndarray
    symbol_centroids_rgb: np.ndarray
    symbol_centroids_chroma: np.ndarray
    pilot_reference_indices: np.ndarray


@dataclass
class RxFrameObservation:
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


# ============================================================
# 2. Utilidades generales
# ============================================================


def _as_gray(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 2:
        return np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Imagen no soportada: shape={arr.shape}")


def _as_bgr(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    raise ValueError(f"Imagen no soportada: shape={arr.shape}")


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
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
    value = 0
    for bit in np.asarray(bits, dtype=np.uint8).reshape(-1):
        value = (value << 1) | int(bit)
    return value


def _bits_to_bytes_rx(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    try:
        return bytes(_tx("bits_to_bytes")(bits))
    except NameError:
        pad = (-bits.size) % 8
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        return np.packbits(bits).tobytes()


def _csk_symbols_to_bits(symbols: np.ndarray) -> np.ndarray:
    symbols = np.asarray(symbols, dtype=np.int16).reshape(-1)
    if np.any(symbols < 0) or np.any(symbols > 7):
        raise ValueError("Los símbolos 8-CSK deben estar entre 0 y 7.")

    try:
        return np.asarray(_tx("csk_symbols_to_bits")(symbols), dtype=np.uint8)
    except NameError:
        labels = tuple(_safe_tx_constant("GRAY_LABELS", GRAY_LABELS_FALLBACK))
        return np.array(
            [bit for symbol in symbols for bit in labels[int(symbol)]],
            dtype=np.uint8,
        )


def _ideal_marker_centers(
    cfg: Any,
    rectified_cell_size: int,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    origins = list(_tx("marker_origins")(cfg))
    centers = [
        [(col + 3.5) * rectified_cell_size, (row + 3.5) * rectified_cell_size]
        for row, col in origins
    ]
    return np.asarray(centers, dtype=np.float32), origins


def _rgb_to_chromaticity(rgb: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """RGB Nx3 -> cromaticidad Nx2 formada por r y g normalizados."""
    arr = np.asarray(rgb, dtype=np.float32)
    one = arr.ndim == 1
    arr = arr.reshape(-1, 3)
    denom = np.maximum(arr.sum(axis=1, keepdims=True), eps)
    chroma = arr[:, :2] / denom
    return chroma[0] if one else chroma


def _apply_color_calibration(rgb: np.ndarray, calibration: ColorCalibration) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    original_shape = arr.shape
    flat = arr.reshape(-1, 3)
    corrected = flat @ calibration.matrix.T + calibration.bias
    return np.clip(corrected, 0.0, 255.0).reshape(original_shape)


# ============================================================
# 3. Detección de marcadores
# ============================================================


def _sample_7x7_pattern(marker_gray: np.ndarray, margin_frac: float = 0.25) -> np.ndarray:
    marker_gray = _as_gray(marker_gray)
    h, w = marker_gray.shape
    means = np.zeros((7, 7), dtype=np.float32)

    for row in range(7):
        for col in range(7):
            y0 = int((row + margin_frac) * h / 7.0)
            y1 = int((row + 1.0 - margin_frac) * h / 7.0)
            x0 = int((col + margin_frac) * w / 7.0)
            x1 = int((col + 1.0 - margin_frac) * w / 7.0)
            roi = marker_gray[y0:y1, x0:x1]
            means[row, col] = float(np.median(roi)) if roi.size else 0.0

    lo = float(np.percentile(means, 15))
    hi = float(np.percentile(means, 85))
    return (means >= 0.5 * (lo + hi)).astype(np.uint8)


def _classify_finder_pattern(sampled_pattern: np.ndarray) -> Tuple[str, int, int]:
    best_type = "unknown"
    best_error = 10**9
    best_k = 0

    for marker_type, template in (
        ("standard", STANDARD_FINDER_PATTERN),
        ("anchor", ANCHOR_PATTERN),
    ):
        for k in range(4):
            error = int(np.count_nonzero(sampled_pattern != np.rot90(template, k)))
            if error < best_error:
                best_type = marker_type
                best_error = error
                best_k = k

    return best_type, best_error, best_k


def _warp_marker_candidate(gray: np.ndarray, quad: np.ndarray, out_px: int = 140) -> np.ndarray:
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
    """Detecta candidatos fiduciales estándar/ancla en un cuadro de video."""
    gray = _as_gray(frame)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    binary_variants: List[np.ndarray] = []
    for mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, bw = cv2.threshold(blur, 0, 255, mode + cv2.THRESH_OTSU)
        bw = cv2.morphologyEx(
            bw,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
        binary_variants.append(bw)

    img_area = gray.shape[0] * gray.shape[1]
    min_area = 0.00012 * img_area
    max_area = 0.20 * img_area
    candidates: List[MarkerCandidate] = []

    for bw in binary_variants:
        contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            quad = _order_quad_points(approx.reshape(4, 2))
            sides = [
                float(np.linalg.norm(quad[i] - quad[(i + 1) % 4]))
                for i in range(4)
            ]
            if min(sides) < 10:
                continue
            if max(sides) / max(min(sides), 1e-6) > 2.3:
                continue

            marker_warp = _warp_marker_candidate(gray_eq, quad, out_px=140)
            sampled = _sample_7x7_pattern(marker_warp)
            marker_type, error, rot_k = _classify_finder_pattern(sampled)
            if error > max_pattern_errors:
                continue

            candidates.append(MarkerCandidate(
                center=tuple(np.mean(quad, axis=0).astype(float)),
                quad=quad,
                area=area,
                marker_type=marker_type,
                pattern_error=error,
                rotation_k=rot_k,
                sampled_pattern=sampled,
            ))

    candidates.sort(key=lambda c: (c.pattern_error, 0 if c.marker_type == "anchor" else 1, -c.area))

    selected: List[MarkerCandidate] = []
    for candidate in candidates:
        duplicate = False
        for previous in selected:
            distance = float(np.hypot(
                candidate.center[0] - previous.center[0],
                candidate.center[1] - previous.center[1],
            ))
            size_ref = 0.5 * (math.sqrt(candidate.area) + math.sqrt(previous.area))
            if distance < 0.5 * size_ref:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)
        if len(selected) >= max_return:
            break

    if debug:
        print(f"Marcadores candidatos: {len(selected)}")
        for i, marker in enumerate(selected):
            print(
                f"  {i}: {marker.marker_type}, error={marker.pattern_error}, "
                f"centro=({marker.center[0]:.1f}, {marker.center[1]:.1f}), "
                f"área={marker.area:.1f}"
            )

    return selected


def draw_marker_overlay(frame: np.ndarray, markers: Sequence[MarkerCandidate]) -> np.ndarray:
    out = _as_bgr(frame).copy()
    for i, marker in enumerate(markers):
        cv2.polylines(out, [marker.quad.astype(int)], True, (0, 255, 0), 2)
        cx, cy = map(int, marker.center)
        cv2.putText(
            out,
            f"{i}:{marker.marker_type},e={marker.pattern_error}",
            (cx - 45, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return out


# ============================================================
# 4. Muestreo RGB, calibración afín y clasificación 8-CSK
# ============================================================


def _sample_cell_rgb(
    warped_bgr: np.ndarray,
    row: int,
    col: int,
    cell_px: int,
    margin_frac: float = 0.28,
) -> np.ndarray:
    """Muestra la región central de una celda y retorna mediana en orden RGB."""
    if not (0.0 <= margin_frac < 0.5):
        raise ValueError("margin_frac debe estar en [0, 0.5).")

    y0 = max(0, int((row + margin_frac) * cell_px))
    y1 = min(warped_bgr.shape[0], int((row + 1.0 - margin_frac) * cell_px))
    x0 = max(0, int((col + margin_frac) * cell_px))
    x1 = min(warped_bgr.shape[1], int((col + 1.0 - margin_frac) * cell_px))
    roi = warped_bgr[y0:y1, x0:x1]

    if roi.size == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    bgr = np.median(roi.reshape(-1, 3), axis=0).astype(np.float32)
    return bgr[::-1].copy()


def _fit_affine_color_model(
    observed_rgb: np.ndarray,
    expected_rgb: np.ndarray,
    robust_iterations: int = 3,
    outlier_sigma: float = 3.5,
    min_inliers: int = 12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Ajusta expected ~= observed @ A.T + b mediante mínimos cuadrados robustos.
    """
    observed = np.asarray(observed_rgb, dtype=np.float64).reshape(-1, 3)
    expected = np.asarray(expected_rgb, dtype=np.float64).reshape(-1, 3)

    valid = np.all(np.isfinite(observed), axis=1) & np.all(np.isfinite(expected), axis=1)
    if int(valid.sum()) < 4:
        raise ValueError("No hay suficientes pilotos válidos para estimar la matriz de color.")

    inliers = valid.copy()
    design_all = np.column_stack([observed, np.ones(observed.shape[0])])

    for _ in range(max(1, robust_iterations)):
        if int(inliers.sum()) < max(4, min_inliers):
            inliers = valid.copy()

        coefficients, *_ = np.linalg.lstsq(
            design_all[inliers],
            expected[inliers],
            rcond=None,
        )  # shape (4, 3)

        predicted = design_all @ coefficients
        residual = np.linalg.norm(predicted - expected, axis=1)
        valid_residual = residual[valid]
        median = float(np.median(valid_residual))
        mad = float(np.median(np.abs(valid_residual - median)))
        robust_scale = max(1.4826 * mad, 1.0)
        threshold = median + outlier_sigma * robust_scale
        new_inliers = valid & (residual <= threshold)

        if int(new_inliers.sum()) < max(4, min_inliers):
            break
        if np.array_equal(new_inliers, inliers):
            inliers = new_inliers
            break
        inliers = new_inliers

    coefficients, *_ = np.linalg.lstsq(
        design_all[inliers],
        expected[inliers],
        rcond=None,
    )
    predicted = design_all @ coefficients
    rmse = float(np.sqrt(np.mean((predicted[inliers] - expected[inliers]) ** 2)))

    matrix = coefficients[:3, :].T.astype(np.float32)
    bias = coefficients[3, :].astype(np.float32)
    return matrix, bias, inliers, rmse


def estimate_color_calibration(
    warped_grid: np.ndarray,
    cfg: Any,
    rectified_cell_size: Optional[int] = None,
    margin_frac: float = 0.28,
    robust_iterations: int = 3,
) -> ColorCalibration:
    """Estima matriz 3x3, sesgo y centroides 8-CSK usando pilotos del frame."""
    cell_px = int(rectified_cell_size or cfg.cell_size)
    warped_bgr = _as_bgr(warped_grid)
    table = list(_tx("get_pilot_reference_table")(cfg))

    observed: List[np.ndarray] = []
    expected: List[Tuple[int, int, int]] = []
    reference_indices: List[int] = []

    for item in table:
        observed.append(_sample_cell_rgb(
            warped_bgr,
            int(item["row"]),
            int(item["col"]),
            cell_px,
            margin_frac,
        ))
        expected.append(tuple(int(v) for v in item["rgb"]))
        reference_indices.append(int(item["reference_index"]))

    observed_rgb = np.asarray(observed, dtype=np.float32)
    expected_rgb = np.asarray(expected, dtype=np.float32)
    ref_indices = np.asarray(reference_indices, dtype=np.int16)

    matrix, bias, inliers, rmse = _fit_affine_color_model(
        observed_rgb,
        expected_rgb,
        robust_iterations=robust_iterations,
        min_inliers=min(12, max(4, len(table) // 3)),
    )

    provisional = ColorCalibration(
        matrix=matrix,
        bias=bias,
        rmse_rgb=rmse,
        inlier_mask=inliers,
        observed_pilots_rgb=observed_rgb,
        expected_pilots_rgb=expected_rgb,
        corrected_pilots_rgb=np.empty_like(observed_rgb),
        symbol_centroids_rgb=np.empty((8, 3), dtype=np.float32),
        symbol_centroids_chroma=np.empty((8, 2), dtype=np.float32),
        pilot_reference_indices=ref_indices,
    )

    corrected = _apply_color_calibration(observed_rgb, provisional).astype(np.float32)
    ideal_palette = np.asarray(cfg.csk_palette_rgb, dtype=np.float32)
    centroids_rgb = np.empty((8, 3), dtype=np.float32)

    for symbol in range(8):
        mask = (ref_indices == symbol) & inliers
        if np.any(mask):
            measured = np.median(corrected[mask], axis=0)
            # Mezcla leve con el ideal para estabilizar pocos pilotos o compresión fuerte.
            centroids_rgb[symbol] = 0.8 * measured + 0.2 * ideal_palette[symbol]
        else:
            centroids_rgb[symbol] = ideal_palette[symbol]

    provisional.corrected_pilots_rgb = corrected
    provisional.symbol_centroids_rgb = centroids_rgb
    provisional.symbol_centroids_chroma = _rgb_to_chromaticity(centroids_rgb).astype(np.float32)
    return provisional


def calibrate_and_sample_csk_symbols(
    warped_grid: np.ndarray,
    cfg: Any,
    rectified_cell_size: Optional[int] = None,
    margin_frac: float = 0.28,
) -> Dict[str, Any]:
    """Calibra color, muestrea DATA y clasifica los símbolos 0..7 por distancia."""
    cell_px = int(rectified_cell_size or cfg.cell_size)
    warped_bgr = _as_bgr(warped_grid)
    calibration = estimate_color_calibration(
        warped_bgr,
        cfg,
        rectified_cell_size=cell_px,
        margin_frac=margin_frac,
    )

    positions = list(_tx("data_positions")(cfg))
    raw_rgb = np.asarray([
        _sample_cell_rgb(warped_bgr, row, col, cell_px, margin_frac)
        for row, col in positions
    ], dtype=np.float32)

    corrected_rgb = _apply_color_calibration(raw_rgb, calibration).astype(np.float32)
    data_chroma = _rgb_to_chromaticity(corrected_rgb).astype(np.float32)
    centroids = calibration.symbol_centroids_chroma

    distances = np.linalg.norm(
        data_chroma[:, None, :] - centroids[None, :, :],
        axis=2,
    ).astype(np.float32)

    order = np.argsort(distances, axis=1)
    symbols = order[:, 0].astype(np.uint8)
    second_symbols = order[:, 1].astype(np.uint8)
    nearest = distances[np.arange(distances.shape[0]), symbols]
    second = distances[np.arange(distances.shape[0]), second_symbols]

    # Confianza relativa: 1 cuando el segundo centroide está mucho más lejos.
    confidence = np.clip((second - nearest) / np.maximum(second, 1e-6), 0.0, 1.0)

    return {
        "symbols": symbols,
        "second_symbols": second_symbols,
        "raw_rgb": raw_rgb,
        "corrected_rgb": corrected_rgb,
        "chromaticity": data_chroma,
        "distances": distances,
        "nearest_distance": nearest,
        "second_distance": second,
        "confidence": confidence.astype(np.float32),
        "calibration": calibration,
        "data_positions": positions,
    }


# ============================================================
# 5. Paquete Fase C y CRC
# ============================================================


def _parse_packet_bits(bits: np.ndarray, encoding: str = "utf-8") -> Dict[str, Any]:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    preamble = np.asarray(
        _safe_tx_constant("PREAMBLE_BITS", PREAMBLE_BITS_FALLBACK),
        dtype=np.uint8,
    ).reshape(-1)
    crc16_ccitt = _tx("crc16_ccitt")

    protocol_expected = int(_safe_tx_constant("PROTOCOL_VERSION", PROTOCOL_VERSION))
    header_len = int(_safe_tx_constant("HEADER_BITS_NO_PAYLOAD", HEADER_BITS_NO_PAYLOAD))
    crc_bits = int(_safe_tx_constant("FRAME_CRC_BITS", FRAME_CRC_BITS))
    frame_type_names = dict(_safe_tx_constant("FRAME_TYPE_NAMES", FRAME_TYPE_NAMES))

    result: Dict[str, Any] = {
        "crc_ok": False,
        "structure_ok": False,
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
        "error": None,
        "score": 10**9,
    }

    if bits.size < header_len + crc_bits:
        result["error"] = "No hay suficientes bits para cabecera y CRC."
        return result

    preamble_errors = int(np.count_nonzero(bits[:preamble.size] != preamble))
    protocol_version = _bits_to_int(bits[32:40])
    frame_type = _bits_to_int(bits[40:48])
    sequence = _bits_to_int(bits[48:64])
    total_data_frames = _bits_to_int(bits[64:80])
    payload_len = _bits_to_int(bits[80:96])
    message_len_bytes = _bits_to_int(bits[96:112])
    message_crc16 = _bits_to_int(bits[112:128])

    max_payload = bits.size - header_len - crc_bits
    packet_len = header_len + payload_len + crc_bits
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
    if protocol_version != protocol_expected:
        structure_penalty += 400
    if frame_type not in frame_type_names:
        structure_penalty += 400
    if total_data_frames <= 0:
        structure_penalty += 300

    if payload_len < 0 or payload_len > max_payload:
        result["error"] = (
            f"payload_len_bits inválido: {payload_len}; máximo posible: {max_payload}."
        )
        result["score"] = 5000 + 50 * preamble_errors + structure_penalty
        return result

    payload_start = header_len
    payload_end = payload_start + payload_len
    crc_start = payload_end
    crc_end = crc_start + crc_bits

    protected_bits = bits[:payload_end]
    payload_bits = bits[payload_start:payload_end]
    rx_crc = _bits_to_int(bits[crc_start:crc_end])
    calc_crc = int(crc16_ccitt(_bits_to_bytes_rx(protected_bits)))

    crc_ok = (
        rx_crc == calc_crc
        and preamble_errors == 0
        and protocol_version == protocol_expected
        and frame_type in frame_type_names
        and total_data_frames > 0
    )

    payload_bytes = _bits_to_bytes_rx(payload_bits)
    n_full_bytes = payload_len // 8
    try:
        partial_text = payload_bytes[:n_full_bytes].decode(encoding, errors="replace")
    except Exception:
        partial_text = ""

    result.update({
        "crc_ok": bool(crc_ok),
        "structure_ok": True,
        "text": partial_text,
        "payload_bits": payload_bits,
        "rx_crc": rx_crc,
        "calc_crc": calc_crc,
        "error": None if crc_ok else "CRC, preámbulo o cabecera inválidos.",
        "score": preamble_errors * 50 + structure_penalty + (0 if crc_ok else 800),
    })
    return result


def decode_warped_grid(
    warped_grid: np.ndarray,
    cfg: Any,
    rectified_cell_size: Optional[int] = None,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """Decodifica una grilla rectificada exclusivamente como 8-CSK."""
    sampled = calibrate_and_sample_csk_symbols(
        warped_grid,
        cfg,
        rectified_cell_size=rectified_cell_size,
    )
    bits = _csk_symbols_to_bits(sampled["symbols"])
    parsed = _parse_packet_bits(bits, encoding=encoding)

    mean_conf = float(np.nanmean(sampled["confidence"])) if sampled["confidence"].size else 0.0
    calibration = sampled["calibration"]
    score = (
        int(parsed["score"])
        + int((1.0 - mean_conf) * 120)
        + int(min(calibration.rmse_rgb, 100.0) * 2.0)
    )

    return {
        **parsed,
        "score": score,
        "sampled": sampled,
        "bits": bits,
        "modulation": "8-CSK Gray",
        "mean_symbol_confidence": mean_conf,
        "color_calibration_rmse": calibration.rmse_rgb,
    }


# ============================================================
# 6. Homografía y procesamiento de un cuadro del archivo
# ============================================================


def _generate_marker_combos(
    markers: Sequence[MarkerCandidate],
    max_marker_candidates: int = 8,
    enforce_anchor: bool = True,
):
    n = min(len(markers), max_marker_candidates)
    combos = list(itertools.combinations(range(n), 4))
    if enforce_anchor:
        combos = [
            combo for combo in combos
            if any(markers[i].marker_type == "anchor" for i in combo)
        ]
    return combos


def _decode_with_homography(
    frame: np.ndarray,
    H: np.ndarray,
    cfg: Any,
    cell_px: int,
    encoding: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    dst_w = int(cfg.grid_cols * cell_px)
    dst_h = int(cfg.grid_rows * cell_px)
    warped = cv2.warpPerspective(_as_bgr(frame), H, (dst_w, dst_h))
    decoded = decode_warped_grid(
        warped,
        cfg,
        rectified_cell_size=cell_px,
        encoding=encoding,
    )
    return warped, decoded


def _process_video_frame(
    frame: np.ndarray,
    cfg: Any,
    rectified_cell_size: Optional[int] = None,
    max_marker_candidates: int = 8,
    cached_homography: Optional[np.ndarray] = None,
    force_redetect: bool = False,
    debug: bool = False,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """Procesa un cuadro del video. Función interna; no abre cámara ni archivos."""
    cell_px = int(rectified_cell_size or cfg.cell_size)
    frame_bgr = _as_bgr(frame)

    # Ruta rápida: reutilizar la geometría ya adquirida.
    if cached_homography is not None and not force_redetect:
        try:
            warped, decoded = _decode_with_homography(
                frame_bgr,
                cached_homography,
                cfg,
                cell_px,
                encoding,
            )
            return {
                "ok": bool(decoded.get("crc_ok", False)),
                "error": None if decoded.get("crc_ok", False) else decoded.get("error"),
                "captured": frame_bgr,
                "overlay": None,
                "markers": [],
                "warped": warped,
                "decoded": decoded,
                "H": cached_homography,
                "homography_source": "cache",
            }
        except Exception as exc:
            if debug:
                print(f"Ruta rápida con homografía cacheada falló: {exc}")

    markers = detect_finder_markers(frame_bgr, debug=debug)
    overlay = draw_marker_overlay(frame_bgr, markers)

    if len(markers) < 4:
        return {
            "ok": False,
            "error": f"Se detectaron {len(markers)} marcadores; se requieren al menos 4.",
            "captured": frame_bgr,
            "overlay": overlay,
            "markers": markers,
            "warped": None,
            "decoded": None,
            "H": None,
            "homography_source": "detection_failed",
        }

    ideal_centers, ideal_origins = _ideal_marker_centers(cfg, cell_px)
    top_left_origin = min(ideal_origins, key=lambda rc: (rc[0], rc[1]))
    evaluations: List[Dict[str, Any]] = []

    anchor_exists = any(marker.marker_type == "anchor" for marker in markers)
    passes = [True, False] if anchor_exists else [False]

    for enforce_anchor in passes:
        combos = _generate_marker_combos(
            markers,
            max_marker_candidates=max_marker_candidates,
            enforce_anchor=enforce_anchor,
        )

        for combo in combos:
            combo_markers = [markers[i] for i in combo]
            src = np.asarray([marker.center for marker in combo_markers], dtype=np.float32)
            anchor_positions = [
                i for i, marker in enumerate(combo_markers)
                if marker.marker_type == "anchor"
            ]

            for permutation in itertools.permutations(range(4)):
                if enforce_anchor and anchor_positions:
                    anchor_pos = anchor_positions[0]
                    if ideal_origins[permutation[anchor_pos]] != top_left_origin:
                        continue

                dst = ideal_centers[list(permutation)].astype(np.float32)
                try:
                    H = cv2.getPerspectiveTransform(src, dst)
                    warped, decoded = _decode_with_homography(
                        frame_bgr,
                        H,
                        cfg,
                        cell_px,
                        encoding,
                    )
                except (cv2.error, ValueError, np.linalg.LinAlgError):
                    continue

                marker_penalty = 3 * sum(marker.pattern_error for marker in combo_markers)
                score = int(decoded["score"]) + marker_penalty
                evaluation = {
                    "score": score,
                    "H": H,
                    "combo": combo,
                    "perm": permutation,
                    "warped": warped,
                    "decoded": decoded,
                }
                evaluations.append(evaluation)

                if decoded.get("crc_ok", False):
                    return {
                        "ok": True,
                        "error": None,
                        "captured": frame_bgr,
                        "overlay": overlay,
                        "markers": markers,
                        "warped": warped,
                        "decoded": decoded,
                        "H": H,
                        "combo": combo,
                        "perm": permutation,
                        "homography_source": "detected",
                    }

    if not evaluations:
        return {
            "ok": False,
            "error": "No se pudo construir una homografía utilizable.",
            "captured": frame_bgr,
            "overlay": overlay,
            "markers": markers,
            "warped": None,
            "decoded": None,
            "H": None,
            "homography_source": "no_evaluation",
        }

    best = min(evaluations, key=lambda item: item["score"])
    return {
        "ok": False,
        "error": "No se validó el CRC del cuadro 8-CSK.",
        "captured": frame_bgr,
        "overlay": overlay,
        "markers": markers,
        "warped": best["warped"],
        "decoded": best["decoded"],
        "H": best["H"],
        "combo": best["combo"],
        "perm": best["perm"],
        "homography_source": "best_unvalidated",
    }


# ============================================================
# 7. Acumulador multi-frame
# ============================================================


def _estimate_frame_confidence(decoded: Dict[str, Any]) -> float:
    base = float(decoded.get("mean_symbol_confidence", 0.0) or 0.0)
    rmse = float(decoded.get("color_calibration_rmse", 100.0) or 100.0)
    preamble_errors = int(decoded.get("preamble_errors") or 0)

    penalty = min(rmse / 255.0, 0.4) + 0.04 * preamble_errors
    if not decoded.get("crc_ok", False):
        penalty += 0.5
    return float(np.clip(base - penalty, 0.0, 1.0))


class RxFrameCollector:
    """Acumula cuadros válidos SYNC/DATA/END y verifica el CRC global."""

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

    def add_decoded(
        self,
        decoded: Dict[str, Any],
        timestamp_s: float,
        capture_index: int,
    ) -> Dict[str, Any]:
        if not decoded.get("crc_ok", False):
            self.status.bad_packets_seen += 1
            self.status.last_error = decoded.get("error") or "CRC de frame inválido"
            return {"accepted": False, "reason": "crc_fail", "done": False}

        obs = RxFrameObservation(
            frame_type=int(decoded["frame_type"]),
            frame_type_name=str(decoded["frame_type_name"]),
            sequence=int(decoded["sequence"]),
            total_data_frames=int(decoded["total_data_frames"]),
            payload_len_bits=int(decoded["payload_len_bits"]),
            message_len_bytes=int(decoded["message_len_bytes"]),
            message_crc16=int(decoded["message_crc16"]),
            payload_bits=np.asarray(decoded.get("payload_bits", []), dtype=np.uint8).copy(),
            confidence=_estimate_frame_confidence(decoded),
            timestamp_s=float(timestamp_s),
            capture_index=int(capture_index),
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
            previous = self.sync_frames.get(obs.sequence)
            if previous is None or obs.confidence > previous.confidence:
                self.sync_frames[obs.sequence] = obs
            self.status.state = "RECEIVING"
            return {"accepted": True, "reason": "sync", "done": False}

        if self.require_sync and self.status.state == "WAITING_SYNC":
            self.status.last_error = "DATA/END recibido antes de SYNC"
            return {"accepted": False, "reason": "waiting_sync", "done": False}

        if not self._metadata_matches(obs):
            self.status.last_error = "Metadatos incompatibles con la sesión actual"
            return {"accepted": False, "reason": "metadata_mismatch", "done": False}

        self._adopt_metadata(obs)

        if obs.frame_type == FRAME_TYPE_DATA:
            if not (0 <= obs.sequence < obs.total_data_frames):
                return {"accepted": False, "reason": "sequence_out_of_range", "done": False}

            previous = self.data_frames.get(obs.sequence)
            if previous is None:
                self.data_frames[obs.sequence] = obs
                reason = "new_data"
            elif obs.confidence > previous.confidence:
                self.data_frames[obs.sequence] = obs
                self.status.duplicate_packets_seen += 1
                reason = "duplicate_replaced"
            else:
                self.status.duplicate_packets_seen += 1
                reason = "duplicate_ignored"

            return {"accepted": True, "reason": reason, "done": self.try_finalize() is not None}

        if obs.frame_type == FRAME_TYPE_END:
            self.status.end_seen += 1
            previous = self.end_frames.get(obs.sequence)
            if previous is None or obs.confidence > previous.confidence:
                self.end_frames[obs.sequence] = obs
            return {"accepted": True, "reason": "end", "done": self.try_finalize() is not None}

        return {"accepted": False, "reason": "unknown_type", "done": False}

    def missing_sequences(self) -> List[int]:
        if self.total_data_frames is None:
            return []
        return [i for i in range(self.total_data_frames) if i not in self.data_frames]

    def try_finalize(self) -> Optional[Dict[str, Any]]:
        if self.final_result is not None:
            return self.final_result
        if self.total_data_frames is None:
            return None
        if self.require_end and self.status.end_seen == 0:
            return None
        if self.missing_sequences():
            return None

        payload_bits = np.concatenate([
            self.data_frames[i].payload_bits
            for i in range(self.total_data_frames)
        ]).astype(np.uint8)

        all_bytes = _bits_to_bytes_rx(payload_bits)
        message_len = int(self.message_len_bytes or 0)
        message_bytes = all_bytes[:message_len]
        calc_crc = int(_tx("crc16_ccitt")(message_bytes))
        expected_crc = int(self.message_crc16 or 0)
        crc_ok = calc_crc == expected_crc

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
            "message_crc16_calc": calc_crc,
            "message_crc_ok": bool(crc_ok),
            "total_data_frames": self.total_data_frames,
            "received_data_frames": len(self.data_frames),
            "missing_sequences": [],
            "sync_seen": self.status.sync_seen,
            "end_seen": self.status.end_seen,
            "status": self.status,
            "data_frames": self.data_frames,
        }
        self.status.state = "DONE" if crc_ok else "CRC_GLOBAL_FAIL"
        return self.final_result

    def snapshot(self) -> Dict[str, Any]:
        return {
            "state": self.status.state,
            "sync_seen": self.status.sync_seen,
            "end_seen": self.status.end_seen,
            "valid_packets_seen": self.status.valid_packets_seen,
            "duplicate_packets_seen": self.status.duplicate_packets_seen,
            "bad_packets_seen": self.status.bad_packets_seen,
            "received_data_frames": len(self.data_frames),
            "total_data_frames": self.total_data_frames,
            "missing_sequences": self.missing_sequences(),
            "last_valid_type": self.status.last_valid_type,
            "last_valid_sequence": self.status.last_valid_sequence,
            "last_error": self.status.last_error,
        }


# ============================================================
# 8. Única entrada de alto nivel: archivo de video
# ============================================================


def _frame_signature(frame: np.ndarray, size: Tuple[int, int] = (96, 54)) -> np.ndarray:
    gray = _as_gray(frame)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32)


def rx_from_video_file(
    path: str | Path,
    cfg: Any,
    rectified_cell_size: Optional[int] = None,
    require_sync: bool = True,
    require_end: bool = True,
    process_every_n: int = 1,
    max_seconds: Optional[float] = None,
    max_marker_candidates: int = 8,
    reuse_homography: bool = True,
    redetect_every_n: int = 0,
    redetect_after_cached_failures: int = 3,
    skip_similar_frames: bool = False,
    similar_frame_threshold: float = 0.8,
    debug: bool = False,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Decodifica una transmisión 8-CSK desde un video ya capturado.

    Esta es la única función de recepción de alto nivel del módulo. No abre la
    cámara ni muestra una vista previa en tiempo real.

    Parámetros principales
    ----------------------
    path:
        Archivo MP4/AVI/MOV previamente capturado.
    process_every_n:
        Procesa uno de cada N cuadros del archivo. Para máxima robustez usa 1.
    max_seconds:
        Límite sobre el TIEMPO DEL VIDEO, no sobre el tiempo de cómputo.
    reuse_homography:
        Tras adquirir una geometría válida, la reutiliza en cuadros posteriores.
    redetect_every_n:
        Si es > 0, vuelve a detectar marcadores cada N cuadros procesados. Es útil
        cuando la cámara o la pantalla se movieron durante la grabación.
    redetect_after_cached_failures:
        Descarta la homografía cacheada después de esta cantidad de fallos CRC
        consecutivos obtenidos con ella. Usa 0 para desactivar esta recuperación.
    skip_similar_frames:
        Omite cuadros casi idénticos al último cuadro procesado. Puede acelerar
        videos con muchas repeticiones, pero conviene dejarlo False al inicio.

    Retorna
    -------
    dict con texto, CRC global, secuencias recibidas, estadísticas del video,
    última decodificación y homografía final.
    """
    if process_every_n < 1:
        raise ValueError("process_every_n debe ser >= 1.")
    if redetect_every_n < 0:
        raise ValueError("redetect_every_n debe ser >= 0.")
    if redetect_after_cached_failures < 0:
        raise ValueError("redetect_after_cached_failures debe ser >= 0.")
    if similar_frame_threshold < 0:
        raise ValueError("similar_frame_threshold debe ser >= 0.")

    video_path = Path(path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    reported_duration = reported_frames / fps if reported_frames > 0 else None

    collector = RxFrameCollector(
        require_sync=require_sync,
        require_end=require_end,
        encoding=encoding,
    )

    start_wall = time.perf_counter()
    capture_index = 0
    processed_count = 0
    skipped_by_stride = 0
    skipped_similar = 0
    valid_decodes = 0
    crc_failures = 0
    exceptions = 0
    full_detections = 0
    cached_decodes = 0
    forced_reacquisitions = 0
    consecutive_cached_failures = 0
    valid_add_events: List[Dict[str, Any]] = []
    last_result: Optional[Dict[str, Any]] = None
    cached_H: Optional[np.ndarray] = None
    last_signature: Optional[np.ndarray] = None
    last_video_time_s = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            capture_index += 1
            video_time_s = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if not np.isfinite(video_time_s) or video_time_s <= 0:
                video_time_s = (capture_index - 1) / fps
            last_video_time_s = video_time_s

            if max_seconds is not None and video_time_s > max_seconds:
                break

            if capture_index % process_every_n != 0:
                skipped_by_stride += 1
                continue

            if skip_similar_frames:
                signature = _frame_signature(frame)
                if last_signature is not None:
                    difference = float(np.mean(np.abs(signature - last_signature)))
                    if difference < similar_frame_threshold:
                        skipped_similar += 1
                        continue
                last_signature = signature

            processed_count += 1
            force_redetect = (
                cached_H is None
                or not reuse_homography
                or (redetect_every_n > 0 and processed_count % redetect_every_n == 0)
            )

            try:
                last_result = _process_video_frame(
                    frame,
                    cfg,
                    rectified_cell_size=rectified_cell_size,
                    max_marker_candidates=max_marker_candidates,
                    cached_homography=cached_H if reuse_homography else None,
                    force_redetect=force_redetect,
                    debug=False,
                    encoding=encoding,
                )

                if last_result.get("homography_source") == "cache":
                    cached_decodes += 1
                else:
                    full_detections += 1

                decoded = last_result.get("decoded")
                if decoded is not None and decoded.get("crc_ok", False):
                    valid_decodes += 1
                    consecutive_cached_failures = 0
                    if reuse_homography and last_result.get("H") is not None:
                        cached_H = np.asarray(last_result["H"], dtype=np.float64)

                    add_info = collector.add_decoded(
                        decoded,
                        timestamp_s=video_time_s,
                        capture_index=capture_index,
                    )
                    valid_add_events.append({
                        **add_info,
                        "capture_index": capture_index,
                        "video_time_s": video_time_s,
                        "frame_type": decoded.get("frame_type_name"),
                        "sequence": decoded.get("sequence"),
                        "confidence": _estimate_frame_confidence(decoded),
                        "calibration_rmse": decoded.get("color_calibration_rmse"),
                    })

                    if debug:
                        print(
                            f"video_frame={capture_index:5d} t={video_time_s:7.3f}s "
                            f"{decoded.get('frame_type_name')} seq={decoded.get('sequence')} "
                            f"conf={_estimate_frame_confidence(decoded):.3f} "
                            f"RMSE={decoded.get('color_calibration_rmse', float('nan')):.2f} "
                            f"-> {add_info['reason']}"
                        )

                    if add_info.get("done"):
                        break
                else:
                    crc_failures += 1
                    if last_result.get("homography_source") == "cache":
                        consecutive_cached_failures += 1
                        if (
                            redetect_after_cached_failures > 0
                            and consecutive_cached_failures >= redetect_after_cached_failures
                        ):
                            cached_H = None
                            consecutive_cached_failures = 0
                            forced_reacquisitions += 1
                            if debug:
                                print("Se descartó la homografía cacheada para readquirir marcadores.")

            except Exception as exc:
                exceptions += 1
                collector.status.bad_packets_seen += 1
                collector.status.last_error = str(exc)
                if debug:
                    print(f"video_frame={capture_index}: error RX: {exc}")

        final = collector.try_finalize()
        if final is None:
            final = {
                "ok": False,
                "text": "",
                "text_best_effort": "",
                "message_crc_ok": False,
                "message_crc16_expected": collector.message_crc16,
                "message_crc16_calc": None,
                "missing_sequences": collector.missing_sequences(),
                "received_data_frames": len(collector.data_frames),
                "total_data_frames": collector.total_data_frames,
                "sync_seen": collector.status.sync_seen,
                "end_seen": collector.status.end_seen,
                "status": collector.status,
                "data_frames": collector.data_frames,
                "snapshot": collector.snapshot(),
            }

        wall_elapsed = time.perf_counter() - start_wall
        final.update({
            "source_video": str(video_path),
            "video_fps": fps,
            "video_reported_frames": reported_frames,
            "video_reported_duration_s": reported_duration,
            "video_time_processed_s": last_video_time_s,
            "processing_elapsed_s": wall_elapsed,
            "capture_frames_seen": capture_index,
            "processed_frames": processed_count,
            "skipped_by_stride": skipped_by_stride,
            "skipped_similar_frames": skipped_similar,
            "valid_frame_decodes": valid_decodes,
            "crc_failed_decodes": crc_failures,
            "processing_exceptions": exceptions,
            "full_marker_detections": full_detections,
            "cached_homography_decodes": cached_decodes,
            "forced_homography_reacquisitions": forced_reacquisitions,
            "valid_add_events": valid_add_events,
            "last_result": last_result,
            "collector": collector,
            "homography": cached_H,
            "modulation": "8-CSK Gray",
        })

        if debug:
            if final.get("ok"):
                print("Recepción 8-CSK desde video exitosa.")
            else:
                print("Recepción 8-CSK incompleta o CRC global fallido.")
            print(
                f"DATA: {final.get('received_data_frames')}/"
                f"{final.get('total_data_frames')} | faltantes={final.get('missing_sequences')}"
            )
            print(
                f"SYNC={final.get('sync_seen')} END={final.get('end_seen')} "
                f"procesados={processed_count}/{capture_index}"
            )

        return final

    finally:
        cap.release()


# ============================================================
# 9. Diagnóstico posterior, sin reproducción en vivo
# ============================================================


def show_last_video_diagnostics(result: Dict[str, Any]) -> None:
    """Muestra después del procesamiento el último cuadro, overlay y grilla."""
    if plt is None:
        raise RuntimeError("matplotlib no está disponible.")

    last = result.get("last_result") or {}
    captured = last.get("captured")
    overlay = last.get("overlay")
    warped = last.get("warped")
    decoded = last.get("decoded") or {}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if captured is not None:
        axes[0].imshow(cv2.cvtColor(_as_bgr(captured), cv2.COLOR_BGR2RGB))
    axes[0].set_title("Último cuadro procesado")
    axes[0].axis("off")

    if overlay is not None:
        axes[1].imshow(cv2.cvtColor(_as_bgr(overlay), cv2.COLOR_BGR2RGB))
    axes[1].set_title("Marcadores")
    axes[1].axis("off")

    if warped is not None:
        axes[2].imshow(cv2.cvtColor(_as_bgr(warped), cv2.COLOR_BGR2RGB))
    axes[2].set_title("Grilla rectificada")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()

    print("========== Diagnóstico RX 8-CSK ==========")
    print(f"Resultado global: {result.get('ok')}")
    print(f"Texto: {result.get('text', '')}")
    print(f"DATA: {result.get('received_data_frames')}/{result.get('total_data_frames')}")
    print(f"Faltantes: {result.get('missing_sequences')}")
    print(f"SYNC: {result.get('sync_seen')} | END: {result.get('end_seen')}")
    if decoded:
        print(f"Último tipo: {decoded.get('frame_type_name')}")
        print(f"Última secuencia: {decoded.get('sequence')}")
        print(f"CRC último frame: {decoded.get('crc_ok')}")
        print(f"Confianza media: {decoded.get('mean_symbol_confidence')}")
        print(f"RMSE calibración RGB: {decoded.get('color_calibration_rmse')}")


def plot_last_constellation(result: Dict[str, Any], max_points: int = 3000) -> None:
    """Grafica la cromaticidad de datos y centroides del último frame procesado."""
    if plt is None:
        raise RuntimeError("matplotlib no está disponible.")

    decoded = ((result.get("last_result") or {}).get("decoded") or {})
    sampled = decoded.get("sampled") or {}
    chroma = sampled.get("chromaticity")
    calibration = sampled.get("calibration")

    if chroma is None or calibration is None:
        raise ValueError("El resultado no contiene una constelación decodificada.")

    points = np.asarray(chroma, dtype=np.float32)
    if points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, max_points).astype(int)
        points = points[indices]

    centroids = np.asarray(calibration.symbol_centroids_chroma, dtype=np.float32)
    plt.figure(figsize=(7, 6))
    plt.scatter(points[:, 0], points[:, 1], s=8, alpha=0.35, label="Celdas DATA")
    plt.scatter(centroids[:, 0], centroids[:, 1], s=110, marker="x", label="Centroides")
    for symbol, (r, g) in enumerate(centroids):
        plt.text(float(r), float(g), f" {symbol}")
    plt.xlabel("r = R/(R+G+B)")
    plt.ylabel("g = G/(R+G+B)")
    plt.title("Constelación recibida 8-CSK")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.show()


__all__ = [
    "bind_tx_helpers",
    "bind_tx_module",
    "rx_from_video_file",
    "show_last_video_diagnostics",
    "plot_last_constellation",
    "detect_finder_markers",
    "draw_marker_overlay",
    "estimate_color_calibration",
    "calibrate_and_sample_csk_symbols",
    "decode_warped_grid",
    "RxFrameCollector",
    "RxFrameObservation",
    "RxCollectorStatus",
    "MarkerCandidate",
    "ColorCalibration",
]
