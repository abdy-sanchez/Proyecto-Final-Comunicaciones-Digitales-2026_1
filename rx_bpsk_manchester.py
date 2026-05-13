"""
rx_bpsk_manchester.py

Receptor/checkpoint para el módem óptico visual BPSK + Manchester.

Este módulo decodifica imágenes estáticas generadas por tx_bpsk_manchester.py.
Está pensado para usarse desde Jupyter Notebook durante el primer punto de control:

    from rx_bpsk_manchester import *

    cfg = RxVisualConfig()
    result = decode_image_file("outputs/fase1/frames/tx_frame_000.png", cfg)
    print(result.text)

Alcance actual:
- Decodifica frames PNG generados digitalmente por el transmisor.
- También puede servir para capturas manuales si la imagen ya está rectificada,
  centrada y con la misma geometría de grilla usada en TX.
- Usa pilotos de brillo para estimar umbral bajo/alto.
- Usa decodificación Manchester diferencial para mayor robustez.
- Verifica preámbulo y CRC16 por frame.

Lo que se deja para la siguiente fase:
- detección automática de pantalla en cámara,
- estimación de homografía/perspectiva,
- compensación fuerte de rotación, escala y rolling shutter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuración compatible con el TX de Fase 1
# ============================================================

@dataclass
class RxVisualConfig:
    # Debe coincidir con TxVisualConfig para esta primera prueba sin canal.
    frame_width: int = 1280
    frame_height: int = 720
    grid_cols: int = 48
    grid_rows: int = 26
    cell_size: int = 24

    gray_low: int = 64
    gray_high: int = 192
    background_gray: int = 16
    quiet_gray: int = 16

    fiducial_low: int = 0
    fiducial_high: int = 255
    marker_size_cells: int = 7

    # Fracción central de cada celda que se promedia.
    # Evita bordes borrosos o mezclados entre celdas.
    sample_fraction: float = 0.55

    # Si True, imprime métricas del proceso.
    verbose: bool = True


DATA = "DATA"
FIDUCIAL = "FIDUCIAL"
QUIET = "QUIET"
PILOT_LOW = "PILOT_LOW"
PILOT_HIGH = "PILOT_HIGH"

PREAMBLE_BITS = np.array(
    [int(b) for b in "10111000110100101110001001011010"],
    dtype=np.uint8,
)


# ============================================================
# Estructura espacial de la grilla
# ============================================================

def marker_origins(cfg: RxVisualConfig) -> List[Tuple[int, int]]:
    """Devuelve las esquinas superiores izquierdas de los cuatro marcadores."""
    m = cfg.marker_size_cells
    return [
        (0, 0),
        (0, cfg.grid_cols - m),
        (cfg.grid_rows - m, 0),
        (cfg.grid_rows - m, cfg.grid_cols - m),
    ]


def build_role_grid(cfg: RxVisualConfig) -> np.ndarray:
    """
    Construye la misma matriz de roles usada por el transmisor:
    DATA, FIDUCIAL, QUIET, PILOT_LOW, PILOT_HIGH.
    """
    roles = np.full((cfg.grid_rows, cfg.grid_cols), DATA, dtype=object)
    m = cfg.marker_size_cells

    for r0, c0 in marker_origins(cfg):
        roles[r0:r0 + m, c0:c0 + m] = FIDUCIAL

    for r0, c0 in marker_origins(cfg):
        r_start = max(0, r0 - 1)
        r_end = min(cfg.grid_rows, r0 + m + 1)
        c_start = max(0, c0 - 1)
        c_end = min(cfg.grid_cols, c0 + m + 1)

        region = roles[r_start:r_end, c_start:c_end]
        region[region != FIDUCIAL] = QUIET

    top_pilot_row = m + 1
    bottom_pilot_row = cfg.grid_rows - m - 2

    for c in range(m + 1, cfg.grid_cols - m - 1):
        if roles[top_pilot_row, c] == DATA:
            roles[top_pilot_row, c] = PILOT_HIGH if c % 2 == 0 else PILOT_LOW

        if roles[bottom_pilot_row, c] == DATA:
            roles[bottom_pilot_row, c] = PILOT_LOW if c % 2 == 0 else PILOT_HIGH

    left_pilot_col = m + 1
    right_pilot_col = cfg.grid_cols - m - 2

    for r in range(m + 1, cfg.grid_rows - m - 1):
        if roles[r, left_pilot_col] == DATA:
            roles[r, left_pilot_col] = PILOT_HIGH if r % 2 == 0 else PILOT_LOW

        if roles[r, right_pilot_col] == DATA:
            roles[r, right_pilot_col] = PILOT_LOW if r % 2 == 0 else PILOT_HIGH

    return roles


def data_positions(cfg: RxVisualConfig) -> List[Tuple[int, int]]:
    """Lista de posiciones DATA en orden fila-columna."""
    roles = build_role_grid(cfg)
    return [
        (r, c)
        for r in range(cfg.grid_rows)
        for c in range(cfg.grid_cols)
        if roles[r, c] == DATA
    ]


def pilot_positions(cfg: RxVisualConfig, role_name: str) -> List[Tuple[int, int]]:
    """Lista de posiciones de pilotos PILOT_LOW o PILOT_HIGH."""
    roles = build_role_grid(cfg)
    return [
        (r, c)
        for r in range(cfg.grid_rows)
        for c in range(cfg.grid_cols)
        if roles[r, c] == role_name
    ]


def grid_offsets(cfg: RxVisualConfig, image_shape: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Offset para ubicar la grilla en la imagen.

    En este checkpoint se asume la misma geometría del TX:
    grilla centrada y sin perspectiva.
    """
    h, w = image_shape[:2]
    grid_w = cfg.grid_cols * cfg.cell_size
    grid_h = cfg.grid_rows * cfg.cell_size

    if grid_w > w or grid_h > h:
        raise ValueError(
            f"La grilla {grid_w}x{grid_h} px no cabe en la imagen {w}x{h} px. "
            "Para fotos de cámara se requerirá rectificación/homografía."
        )

    x0 = (w - grid_w) // 2
    y0 = (h - grid_h) // 2
    return x0, y0


# ============================================================
# Utilidades de bits y CRC
# ============================================================

def bits_to_int(bits: Sequence[int]) -> int:
    """Convierte bits MSB-first a entero."""
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """
    Convierte bits MSB-first a bytes.
    Si la longitud no es múltiplo de 8, rellena con ceros al final.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size == 0:
        return b""

    pad = (-bits.size) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

    return np.packbits(bits).tobytes()


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE compatible con el transmisor."""
    crc = init

    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF

    return crc & 0xFFFF


# ============================================================
# Lectura y muestreo de imagen
# ============================================================

def load_grayscale_image(path: str | Path) -> np.ndarray:
    """Carga una imagen como escala de grises uint8."""
    path = Path(path)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")

    return img


def ensure_grayscale_uint8(image: np.ndarray) -> np.ndarray:
    """Convierte una imagen BGR/RGB/gris a gris uint8."""
    img = np.asarray(image)

    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] == 3:
        # OpenCV suele usar BGR; si viene de matplotlib/PIL como RGB, la conversión
        # cambia muy poco en escala de grises para estos patrones neutros.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3 and img.shape[2] == 4:
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Formato de imagen no soportado: shape={img.shape}")

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def sample_cell_mean(
    gray: np.ndarray,
    cfg: RxVisualConfig,
    row: int,
    col: int,
) -> float:
    """
    Promedia el centro de una celda de la grilla.
    """
    if not (0 < cfg.sample_fraction <= 1.0):
        raise ValueError("sample_fraction debe estar en el intervalo (0, 1].")

    x_offset, y_offset = grid_offsets(cfg, gray.shape)
    s = cfg.cell_size

    margin = int(round((1.0 - cfg.sample_fraction) * s / 2.0))

    y0 = y_offset + row * s + margin
    y1 = y_offset + (row + 1) * s - margin
    x0 = x_offset + col * s + margin
    x1 = x_offset + (col + 1) * s - margin

    patch = gray[y0:y1, x0:x1]

    if patch.size == 0:
        raise ValueError(f"Parche vacío en celda ({row}, {col}).")

    return float(np.mean(patch))


def sample_positions(
    gray: np.ndarray,
    cfg: RxVisualConfig,
    positions: Iterable[Tuple[int, int]],
) -> np.ndarray:
    """Muestrea una lista de celdas y retorna intensidades promedio."""
    return np.array(
        [sample_cell_mean(gray, cfg, r, c) for r, c in positions],
        dtype=float,
    )


def estimate_pilot_levels(gray: np.ndarray, cfg: RxVisualConfig) -> Tuple[float, float, float]:
    """
    Estima nivel bajo, nivel alto y umbral usando pilotos de brillo.
    """
    low_pos = pilot_positions(cfg, PILOT_LOW)
    high_pos = pilot_positions(cfg, PILOT_HIGH)

    if not low_pos or not high_pos:
        raise ValueError("No se encontraron pilotos de brillo en la grilla.")

    low_values = sample_positions(gray, cfg, low_pos)
    high_values = sample_positions(gray, cfg, high_pos)

    low_level = float(np.median(low_values))
    high_level = float(np.median(high_values))
    threshold = 0.5 * (low_level + high_level)

    return low_level, high_level, threshold


# ============================================================
# Demodulación BPSK visual y Manchester
# ============================================================

def sample_data_values(gray: np.ndarray, cfg: RxVisualConfig) -> np.ndarray:
    """Extrae la intensidad promedio de todas las celdas DATA en orden TX."""
    return sample_positions(gray, cfg, data_positions(cfg))


def hard_decision_symbols(values: np.ndarray, threshold: float) -> np.ndarray:
    """Convierte intensidades en símbolos 0/1 con umbral adaptativo."""
    return (np.asarray(values) >= threshold).astype(np.uint8)


def manchester_decode_from_symbols(symbols: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Decodifica Manchester usando símbolos ya umbralizados.

    Convención compatible con TX:
        [1, 0] -> bit 0
        [0, 1] -> bit 1

    Retorna:
        bits decodificados,
        número de pares Manchester inválidos.
    """
    symbols = np.asarray(symbols, dtype=np.uint8)
    n_pairs = symbols.size // 2
    pairs = symbols[:2 * n_pairs].reshape(-1, 2)

    bits = np.zeros(n_pairs, dtype=np.uint8)
    invalid = 0

    for i, (a, b) in enumerate(pairs):
        if a == 1 and b == 0:
            bits[i] = 0
        elif a == 0 and b == 1:
            bits[i] = 1
        else:
            # Par inválido: conserva una decisión aproximada.
            # Si el segundo símbolo es mayor que el primero, se parece a bit 1.
            bits[i] = 1 if b > a else 0
            invalid += 1

    return bits, invalid


def manchester_decode_differential(values: np.ndarray) -> np.ndarray:
    """
    Decodificación Manchester diferencial directamente desde intensidades.

    bit 0 -> [alto, bajo], por tanto segundo - primero < 0.
    bit 1 -> [bajo, alto], por tanto segundo - primero > 0.

    Esta versión suele ser más robusta que solo umbralizar, porque cada bit se
    decide por transición local.
    """
    values = np.asarray(values, dtype=float)
    n_pairs = values.size // 2
    pairs = values[:2 * n_pairs].reshape(-1, 2)
    return (pairs[:, 1] > pairs[:, 0]).astype(np.uint8)


# ============================================================
# Parseo de paquete
# ============================================================

@dataclass
class FrameDecodeResult:
    ok: bool
    text: str
    payload_bits: np.ndarray
    frame_index: int
    total_frames: int
    payload_length_bits: int
    crc_received: int
    crc_calculated: int
    crc_ok: bool
    preamble_ok: bool
    manchester_invalid_pairs: int
    low_level: float
    high_level: float
    threshold: float
    message: str


def parse_packet_bits(
    packet_bits: np.ndarray,
    *,
    manchester_invalid_pairs: int,
    low_level: float,
    high_level: float,
    threshold: float,
    encoding: str = "utf-8",
) -> FrameDecodeResult:
    """Interpreta los bits crudos de un frame después de Manchester."""
    packet_bits = np.asarray(packet_bits, dtype=np.uint8)

    min_header_bits = 32 + 16 + 16 + 16 + 16
    if packet_bits.size < min_header_bits:
        return FrameDecodeResult(
            ok=False,
            text="",
            payload_bits=np.array([], dtype=np.uint8),
            frame_index=-1,
            total_frames=-1,
            payload_length_bits=0,
            crc_received=-1,
            crc_calculated=-1,
            crc_ok=False,
            preamble_ok=False,
            manchester_invalid_pairs=manchester_invalid_pairs,
            low_level=low_level,
            high_level=high_level,
            threshold=threshold,
            message="El frame no tiene suficientes bits para encabezado + CRC.",
        )

    preamble = packet_bits[0:32]
    frame_index = bits_to_int(packet_bits[32:48])
    total_frames = bits_to_int(packet_bits[48:64])
    payload_length_bits = bits_to_int(packet_bits[64:80])

    payload_start = 80
    payload_end = payload_start + payload_length_bits
    crc_start = payload_end
    crc_end = crc_start + 16

    if crc_end > packet_bits.size:
        return FrameDecodeResult(
            ok=False,
            text="",
            payload_bits=np.array([], dtype=np.uint8),
            frame_index=frame_index,
            total_frames=total_frames,
            payload_length_bits=payload_length_bits,
            crc_received=-1,
            crc_calculated=-1,
            crc_ok=False,
            preamble_ok=bool(np.array_equal(preamble, PREAMBLE_BITS)),
            manchester_invalid_pairs=manchester_invalid_pairs,
            low_level=low_level,
            high_level=high_level,
            threshold=threshold,
            message=(
                "La longitud de payload indicada en el encabezado excede la "
                "capacidad decodificada del frame."
            ),
        )

    payload_bits = packet_bits[payload_start:payload_end]
    crc_received = bits_to_int(packet_bits[crc_start:crc_end])

    header_and_payload = packet_bits[:payload_end]
    crc_calculated = crc16_ccitt(bits_to_bytes(header_and_payload))

    preamble_ok = bool(np.array_equal(preamble, PREAMBLE_BITS))
    crc_ok = bool(crc_received == crc_calculated)

    text = ""
    decode_msg = ""

    if payload_bits.size > 0:
        try:
            text = bits_to_bytes(payload_bits).decode(encoding)
        except UnicodeDecodeError as exc:
            decode_msg = f"Payload recibido, pero aún no forma texto UTF-8 completo: {exc}"

    ok = preamble_ok and crc_ok

    if ok:
        msg = "Frame decodificado correctamente."
    else:
        problems = []
        if not preamble_ok:
            problems.append("preámbulo no coincide")
        if not crc_ok:
            problems.append("CRC no coincide")
        if manchester_invalid_pairs:
            problems.append(f"{manchester_invalid_pairs} pares Manchester inválidos")
        if decode_msg:
            problems.append(decode_msg)
        msg = "; ".join(problems) if problems else "Frame decodificado con advertencias."

    return FrameDecodeResult(
        ok=ok,
        text=text,
        payload_bits=payload_bits.copy(),
        frame_index=frame_index,
        total_frames=total_frames,
        payload_length_bits=payload_length_bits,
        crc_received=crc_received,
        crc_calculated=crc_calculated,
        crc_ok=crc_ok,
        preamble_ok=preamble_ok,
        manchester_invalid_pairs=manchester_invalid_pairs,
        low_level=low_level,
        high_level=high_level,
        threshold=threshold,
        message=msg,
    )


# ============================================================
# API principal de decodificación
# ============================================================

def decode_frame_image(
    image: np.ndarray,
    cfg: Optional[RxVisualConfig] = None,
    *,
    encoding: str = "utf-8",
    use_differential_manchester: bool = True,
) -> FrameDecodeResult:
    """
    Decodifica un único frame visual.

    Parámetros:
        image: imagen ndarray en gris/BGR/RGB.
        cfg: configuración geométrica compatible con el TX.
        encoding: codificación de texto final.
        use_differential_manchester: si True, decide cada bit por diferencia
            entre las dos mitades Manchester. Recomendado.
    """
    cfg = cfg or RxVisualConfig()
    gray = ensure_grayscale_uint8(image)

    low_level, high_level, threshold = estimate_pilot_levels(gray, cfg)
    data_values = sample_data_values(gray, cfg)
    symbols = hard_decision_symbols(data_values, threshold)
    bits_from_symbols, invalid_pairs = manchester_decode_from_symbols(symbols)

    if use_differential_manchester:
        packet_bits = manchester_decode_differential(data_values)
    else:
        packet_bits = bits_from_symbols

    result = parse_packet_bits(
        packet_bits,
        manchester_invalid_pairs=invalid_pairs,
        low_level=low_level,
        high_level=high_level,
        threshold=threshold,
        encoding=encoding,
    )

    if cfg.verbose:
        print_decode_summary(result)

    return result


def decode_image_file(
    path: str | Path,
    cfg: Optional[RxVisualConfig] = None,
    *,
    encoding: str = "utf-8",
    use_differential_manchester: bool = True,
) -> FrameDecodeResult:
    """Carga y decodifica un frame PNG/JPG."""
    image = load_grayscale_image(path)
    return decode_frame_image(
        image,
        cfg=cfg,
        encoding=encoding,
        use_differential_manchester=use_differential_manchester,
    )


def decode_image_files(
    paths: Sequence[str | Path],
    cfg: Optional[RxVisualConfig] = None,
    *,
    encoding: str = "utf-8",
    use_differential_manchester: bool = True,
) -> Tuple[str, List[FrameDecodeResult]]:
    """
    Decodifica varios frames y reconstruye el texto completo.

    Retorna:
        texto_reconstruido,
        lista de resultados por frame.
    """
    cfg = cfg or RxVisualConfig(verbose=False)
    results = [
        decode_image_file(
            path,
            cfg=cfg,
            encoding=encoding,
            use_differential_manchester=use_differential_manchester,
        )
        for path in paths
    ]

    valid = [r for r in results if r.ok]

    if not valid:
        raise ValueError("Ningún frame pasó preámbulo + CRC.")

    # Ordenar por índice de frame y concatenar payloads.
    valid_sorted = sorted(valid, key=lambda r: r.frame_index)
    payload_bits = np.concatenate([r.payload_bits for r in valid_sorted])

    text = bits_to_bytes(payload_bits).decode(encoding)
    return text, results


def decode_folder(
    folder: str | Path,
    pattern: str = "tx_frame_*.png",
    cfg: Optional[RxVisualConfig] = None,
    *,
    encoding: str = "utf-8",
) -> Tuple[str, List[FrameDecodeResult]]:
    """Decodifica todos los frames de una carpeta en orden alfabético."""
    folder = Path(folder)
    paths = sorted(folder.glob(pattern))

    if not paths:
        raise FileNotFoundError(f"No se encontraron imágenes {pattern} en {folder}")

    return decode_image_files(paths, cfg=cfg, encoding=encoding)


def print_decode_summary(result: FrameDecodeResult) -> None:
    """Imprime resumen legible del resultado de decodificación."""
    print("=== RX BPSK + Manchester ===")
    print(f"OK general: {result.ok}")
    print(f"Preámbulo OK: {result.preamble_ok}")
    print(f"CRC OK: {result.crc_ok}")
    print(f"Frame: {result.frame_index + 1}/{result.total_frames}")
    print(f"Payload: {result.payload_length_bits} bits")
    print(f"Piloto bajo estimado: {result.low_level:.2f}")
    print(f"Piloto alto estimado: {result.high_level:.2f}")
    print(f"Umbral: {result.threshold:.2f}")
    print(f"Pares Manchester inválidos: {result.manchester_invalid_pairs}")
    print(f"Mensaje: {result.message}")
    if result.text:
        print(f"Texto parcial: {result.text!r}")


# ============================================================
# Visualización de diagnóstico para Jupyter
# ============================================================

def plot_cell_histogram(
    image: np.ndarray,
    cfg: Optional[RxVisualConfig] = None,
    bins: int = 40,
) -> None:
    """
    Muestra histograma de intensidades de DATA y pilotos.
    Útil para ver separación entre niveles bajo/alto.
    """
    cfg = cfg or RxVisualConfig(verbose=False)
    gray = ensure_grayscale_uint8(image)

    data_values = sample_data_values(gray, cfg)
    low_values = sample_positions(gray, cfg, pilot_positions(cfg, PILOT_LOW))
    high_values = sample_positions(gray, cfg, pilot_positions(cfg, PILOT_HIGH))
    low_level, high_level, threshold = estimate_pilot_levels(gray, cfg)

    plt.figure(figsize=(8, 4))
    plt.hist(data_values, bins=bins, alpha=0.55, label="DATA")
    plt.hist(low_values, bins=bins, alpha=0.55, label="PILOT_LOW")
    plt.hist(high_values, bins=bins, alpha=0.55, label="PILOT_HIGH")
    plt.axvline(threshold, linestyle="--", label=f"umbral={threshold:.1f}")
    plt.title("Histograma de intensidades muestreadas por celda")
    plt.xlabel("Intensidad promedio")
    plt.ylabel("Conteo")
    plt.legend()
    plt.grid(True)
    plt.show()


def show_sampling_overlay(
    image: np.ndarray,
    cfg: Optional[RxVisualConfig] = None,
    max_points: int = 350,
) -> None:
    """
    Dibuja puntos de muestreo sobre una imagen para verificar alineación.
    """
    cfg = cfg or RxVisualConfig(verbose=False)
    gray = ensure_grayscale_uint8(image)
    x_offset, y_offset = grid_offsets(cfg, gray.shape)
    s = cfg.cell_size

    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    positions = data_positions(cfg)

    step = max(1, len(positions) // max_points)
    for r, c in positions[::step]:
        x = int(x_offset + (c + 0.5) * s)
        y = int(y_offset + (r + 0.5) * s)
        cv2.circle(display, (x, y), radius=2, color=(255, 0, 0), thickness=-1)

    plt.figure(figsize=(10, 6))
    plt.imshow(display)
    plt.title("Puntos de muestreo DATA sobre el frame")
    plt.axis("off")
    plt.show()
