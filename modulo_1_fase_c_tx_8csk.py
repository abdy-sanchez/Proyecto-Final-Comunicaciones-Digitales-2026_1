"""
Módulo TX Fase C: texto -> secuencia multi-frame visual 8-CSK.

Este archivo está pensado para importarse desde un notebook. No ejecuta la
transmisión ni abre ventanas al importarlo.

Características principales
---------------------------
- Modulación 8-CSK: cada macropíxel DATA transporta 3 bits.
- Constelación definida en RGB con luminancia CIELAB aproximadamente constante.
- Etiquetado Gray alrededor de la constelación para que colores vecinos difieran
  en un solo bit.
- Pilotos RGB en cada frame: los 8 colores de datos y 3 referencias neutras.
- Mantiene los cuatro marcadores fiduciales y el marcador ancla de orientación.
- Mantiene el protocolo multi-frame: SYNC | DATA... | END.
- Cada frame incluye cabecera, número de secuencia, longitudes y CRC-16.
- Permite guardar PNG, crear un video o transmitir en pantalla completa.

Convención de color
-------------------
Las paletas y funciones públicas reciben colores en orden RGB. Las imágenes
retornadas por OpenCV están en orden BGR, que es la convención nativa de cv2.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from IPython.display import clear_output
except Exception:  # pragma: no cover
    clear_output = None

np.set_printoptions(precision=3, suppress=True)

RGB = Tuple[int, int, int]

# Los ocho puntos fueron obtenidos aproximadamente sobre un círculo de cromaticidad
# en CIELAB con L*=65 y C*=35. De esta forma, la luminancia perceptual se mantiene
# aproximadamente constante mientras cambia principalmente el tono.
#
# El orden recorre el círculo cromático. Los bits asignados siguen un código Gray:
# 000, 001, 011, 010, 110, 111, 101, 100.
DEFAULT_CSK_PALETTE_RGB: Tuple[RGB, ...] = (
    (217, 134, 159),  # símbolo 0, etiqueta 000: rosa/rojo
    (213, 140, 115),  # símbolo 1, etiqueta 001: naranja
    (179, 156, 95),   # símbolo 2, etiqueta 011: amarillo/oliva
    (128, 169, 112),  # símbolo 3, etiqueta 010: verde
    (67, 175, 157),   # símbolo 4, etiqueta 110: verde-cian
    (30, 172, 202),   # símbolo 5, etiqueta 111: cian
    (113, 161, 220),  # símbolo 6, etiqueta 101: azul
    (182, 144, 203),  # símbolo 7, etiqueta 100: magenta
)

GRAY_LABELS: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 1, 1),
    (0, 1, 0),
    (1, 1, 0),
    (1, 1, 1),
    (1, 0, 1),
    (1, 0, 0),
)

BITS_PER_SYMBOL = 3


@dataclass
class TxVisualConfig:
    # Tamaño del frame generado.
    frame_width: int = 1280
    frame_height: int = 720

    # Grilla visual.
    grid_cols: int = 48
    grid_rows: int = 26
    cell_size: int = 24

    # Constelación 8-CSK. Siempre se expresa en RGB.
    csk_palette_rgb: Tuple[RGB, ...] = DEFAULT_CSK_PALETTE_RGB

    # Referencias neutras adicionales para que el receptor pueda estimar una
    # corrección afín de color (matriz 3x3 + sesgo) junto con los 8 pilotos CSK.
    neutral_pilots_rgb: Tuple[RGB, ...] = (
        (48, 48, 48),
        (128, 128, 128),
        (208, 208, 208),
    )

    # Fondo y zonas silenciosas, expresados en RGB.
    background_rgb: RGB = (16, 16, 16)
    quiet_rgb: RGB = (16, 16, 16)

    # Marcadores fiduciales monocromáticos, expresados en RGB.
    fiducial_low_rgb: RGB = (0, 0, 0)
    fiducial_high_rgb: RGB = (255, 255, 255)
    marker_size_cells: int = 7

    # Duración por frame visual durante transmisión real.
    frame_duration_s: float = 0.12

    # Estructura temporal de la Fase C.
    sync_frames: int = 3
    end_frames: int = 3

    # Carpetas / archivos de salida.
    output_dir: Path = Path("outputs/fase_c_8csk/frames")
    video_output_path: Path = Path("outputs/fase_c_8csk/tx_sequence_8csk.mp4")

    # Útil para depuración, pero mantener False para transmisión real.
    debug_text: bool = False

    def __post_init__(self) -> None:
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("frame_width y frame_height deben ser positivos.")
        if self.grid_cols <= 0 or self.grid_rows <= 0 or self.cell_size <= 0:
            raise ValueError("grid_cols, grid_rows y cell_size deben ser positivos.")
        if self.marker_size_cells != 7:
            raise ValueError(
                "Este módulo usa patrones fiduciales 7x7; marker_size_cells debe ser 7."
            )
        if len(self.csk_palette_rgb) != 8:
            raise ValueError("8-CSK requiere exactamente 8 colores en csk_palette_rgb.")
        if len(self.neutral_pilots_rgb) < 3:
            raise ValueError("Se recomiendan al menos 3 pilotos neutros.")
        if self.frame_duration_s <= 0:
            raise ValueError("frame_duration_s debe ser positivo.")
        if self.sync_frames < 1 or self.end_frames < 1:
            raise ValueError("sync_frames y end_frames deben ser al menos 1.")

        all_colors = (
            tuple(self.csk_palette_rgb)
            + tuple(self.neutral_pilots_rgb)
            + (
                self.background_rgb,
                self.quiet_rgb,
                self.fiducial_low_rgb,
                self.fiducial_high_rgb,
            )
        )
        for color in all_colors:
            _validate_rgb(color)


@dataclass
class TxFrameRecord:
    """Registro interno de cada frame visual generado."""

    frame_type_name: str
    frame_type_id: int
    sequence: int
    total_data_frames: int
    payload_len_bits: int
    packet_bits: np.ndarray
    symbols: np.ndarray
    symbol_padding_bits: int
    image: np.ndarray


DATA = "DATA"
FIDUCIAL = "FIDUCIAL"
QUIET = "QUIET"
PILOT = "PILOT"

FRAME_TYPE_SYNC = 1
FRAME_TYPE_DATA = 2
FRAME_TYPE_END = 3

FRAME_TYPE_NAMES = {
    FRAME_TYPE_SYNC: "SYNC",
    FRAME_TYPE_DATA: "DATA",
    FRAME_TYPE_END: "END",
}

# Se incrementa respecto al transmisor BPSK/Manchester porque el mapeo físico cambió.
PROTOCOL_VERSION = 2

# Preámbulo fijo de 32 bits. Se conserva para facilitar la comparación con el
# sistema anterior y para que el receptor valide sincronización.
PREAMBLE_BITS = np.array(
    [int(b) for b in "10111000110100101110001001011010"],
    dtype=np.uint8,
)

# Cabecera Fase C, antes del payload:
#   preamble            32 bits
#   protocol_version     8 bits
#   frame_type           8 bits
#   sequence            16 bits
#   total_data_frames   16 bits
#   payload_len_bits    16 bits
#   message_len_bytes   16 bits
#   message_crc16       16 bits
# Después del payload:
#   frame_crc16         16 bits
HEADER_BITS_NO_PAYLOAD = 32 + 8 + 8 + 16 + 16 + 16 + 16 + 16
FRAME_CRC_BITS = 16

def _validate_rgb(color: Sequence[int]) -> None:
    if len(color) != 3:
        raise ValueError(f"Un color RGB debe tener 3 componentes; se recibió {color}.")
    if any(int(v) < 0 or int(v) > 255 for v in color):
        raise ValueError(f"Las componentes RGB deben estar entre 0 y 255; se recibió {color}.")


# Configuración por defecto para notebook.
cfg = TxVisualConfig()


def rgb_to_bgr(rgb: Sequence[int]) -> Tuple[int, int, int]:
    """Convierte una tripleta RGB a BGR para escribirla en una imagen OpenCV."""
    _validate_rgb(rgb)
    r, g, b = (int(v) for v in rgb)
    return b, g, r


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Convierte bytes a bits MSB-first."""
    if not data:
        return np.array([], dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Convierte bits MSB-first a bytes; rellena con ceros si hace falta."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)

    if bits.size == 0:
        return b""
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits_to_bytes solo acepta bits 0/1.")

    pad = (-bits.size) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])

    return np.packbits(bits).tobytes()


def int_to_bits(value: int, nbits: int) -> np.ndarray:
    """Convierte un entero no negativo a nbits MSB-first."""
    if value < 0:
        raise ValueError("value debe ser no negativo.")
    if value >= (1 << nbits):
        raise ValueError(f"value={value} no cabe en {nbits} bits.")

    return np.array(
        [(value >> shift) & 1 for shift in range(nbits - 1, -1, -1)],
        dtype=np.uint8,
    )


def _gray_label_to_symbol_map() -> Dict[Tuple[int, int, int], int]:
    return {label: index for index, label in enumerate(GRAY_LABELS)}


def bits_to_csk_symbols(bits: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Agrupa bits de tres en tres y los mapea a índices de símbolo 8-CSK.

    El mapeo usa las etiquetas Gray de GRAY_LABELS. Si la longitud no es múltiplo
    de tres, se agregan 1 o 2 ceros al final. Retorna los símbolos y la cantidad
    exacta de bits de relleno agregados.
    """
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("bits_to_csk_symbols solo acepta bits 0/1.")

    pad_bits = (-bits.size) % BITS_PER_SYMBOL
    if pad_bits:
        bits_padded = np.concatenate([bits, np.zeros(pad_bits, dtype=np.uint8)])
    else:
        bits_padded = bits

    if bits_padded.size == 0:
        return np.array([], dtype=np.uint8), pad_bits

    groups = bits_padded.reshape(-1, BITS_PER_SYMBOL)
    label_to_symbol = _gray_label_to_symbol_map()
    symbols = np.array(
        [label_to_symbol[tuple(int(v) for v in group)] for group in groups],
        dtype=np.uint8,
    )
    return symbols, pad_bits


def csk_symbols_to_bits(symbols: np.ndarray, trim_padding_bits: int = 0) -> np.ndarray:
    """
    Conversión inversa ideal símbolo -> bits.

    Esta función es útil para pruebas digitales del transmisor. El receptor real
    primero deberá clasificar cada color y obtener los índices 0..7.
    """
    symbols = np.asarray(symbols, dtype=np.int16).reshape(-1)
    if np.any(symbols < 0) or np.any(symbols > 7):
        raise ValueError("Los símbolos 8-CSK deben estar entre 0 y 7.")
    if trim_padding_bits not in (0, 1, 2):
        raise ValueError("trim_padding_bits debe ser 0, 1 o 2.")

    if symbols.size == 0:
        return np.array([], dtype=np.uint8)

    bits = np.array(
        [bit for symbol in symbols for bit in GRAY_LABELS[int(symbol)]],
        dtype=np.uint8,
    )
    if trim_padding_bits:
        bits = bits[:-trim_padding_bits]
    return bits


def crc16_ccitt(data: bytes, init: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE: poly=0x1021, init=0xFFFF."""
    crc = init

    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF

    return crc & 0xFFFF


def marker_origins(cfg: TxVisualConfig) -> List[Tuple[int, int]]:
    """Esquinas superiores izquierdas de los cuatro marcadores."""
    m = cfg.marker_size_cells
    return [
        (0, 0),
        (0, cfg.grid_cols - m),
        (cfg.grid_rows - m, 0),
        (cfg.grid_rows - m, cfg.grid_cols - m),
    ]


def _pilot_ring_positions(cfg: TxVisualConfig) -> List[Tuple[int, int]]:
    """Posiciones de pilotos ordenadas alrededor del perímetro interior."""
    m = cfg.marker_size_cells
    top = m + 1
    bottom = cfg.grid_rows - m - 2
    left = m + 1
    right = cfg.grid_cols - m - 2

    if top >= bottom or left >= right:
        raise ValueError(
            "La grilla es demasiado pequeña para marcadores, zona silenciosa y pilotos."
        )

    positions: List[Tuple[int, int]] = []

    # Superior: izquierda -> derecha.
    positions.extend((top, c) for c in range(left, right + 1))
    # Derecha: arriba -> abajo, sin repetir esquinas.
    positions.extend((r, right) for r in range(top + 1, bottom))
    # Inferior: derecha -> izquierda.
    positions.extend((bottom, c) for c in range(right, left - 1, -1))
    # Izquierda: abajo -> arriba, sin repetir esquinas.
    positions.extend((r, left) for r in range(bottom - 1, top, -1))

    return positions


def pilot_palette_rgb(cfg: TxVisualConfig) -> Tuple[RGB, ...]:
    """Paleta completa conocida por TX/RX: 8 símbolos CSK + neutros."""
    return tuple(cfg.csk_palette_rgb) + tuple(cfg.neutral_pilots_rgb)


def pilot_assignments(cfg: TxVisualConfig) -> List[Tuple[int, int, int]]:
    """
    Retorna (fila, columna, índice_de_referencia) para cada piloto.

    Los índices 0..7 corresponden directamente a los símbolos 8-CSK. Los índices
    siguientes corresponden a los pilotos neutros.
    """
    positions = _pilot_ring_positions(cfg)
    palette_len = len(pilot_palette_rgb(cfg))
    return [
        (row, col, i % palette_len)
        for i, (row, col) in enumerate(positions)
    ]


def get_pilot_reference_table(cfg: TxVisualConfig) -> List[dict]:
    """Tabla que el receptor puede usar para asociar posiciones y RGB esperado."""
    palette = pilot_palette_rgb(cfg)
    table: List[dict] = []
    for row, col, ref_index in pilot_assignments(cfg):
        if ref_index < 8:
            name = f"CSK_{ref_index}_{''.join(map(str, GRAY_LABELS[ref_index]))}"
        else:
            name = f"NEUTRAL_{ref_index - 8}"
        table.append(
            {
                "row": row,
                "col": col,
                "reference_index": ref_index,
                "name": name,
                "rgb": palette[ref_index],
            }
        )
    return table


def build_role_grid(cfg: TxVisualConfig) -> np.ndarray:
    """
    Construye la máscara de roles de la grilla.

    Roles:
    - DATA: símbolos 8-CSK.
    - FIDUCIAL: marcadores tipo finder.
    - QUIET: zona oscura alrededor de marcadores.
    - PILOT: referencias RGB conocidas.
    """
    roles = np.full((cfg.grid_rows, cfg.grid_cols), DATA, dtype=object)
    m = cfg.marker_size_cells

    for r0, c0 in marker_origins(cfg):
        if r0 < 0 or c0 < 0:
            raise ValueError("La grilla es menor que los marcadores fiduciales.")
        roles[r0:r0 + m, c0:c0 + m] = FIDUCIAL

    # Zona silenciosa de una celda alrededor de cada marcador.
    for r0, c0 in marker_origins(cfg):
        r_start = max(0, r0 - 1)
        r_end = min(cfg.grid_rows, r0 + m + 1)
        c_start = max(0, c0 - 1)
        c_end = min(cfg.grid_cols, c0 + m + 1)

        region = roles[r_start:r_end, c_start:c_end]
        region[region != FIDUCIAL] = QUIET

    for row, col, _ in pilot_assignments(cfg):
        if roles[row, col] != DATA:
            raise ValueError(
                f"El piloto ({row}, {col}) colisiona con una región {roles[row, col]}."
            )
        roles[row, col] = PILOT

    return roles


def data_positions(cfg: TxVisualConfig) -> List[Tuple[int, int]]:
    """Lista de posiciones DATA en orden fila-columna."""
    roles = build_role_grid(cfg)
    return [
        (r, c)
        for r in range(cfg.grid_rows)
        for c in range(cfg.grid_cols)
        if roles[r, c] == DATA
    ]


def get_capacity(cfg: TxVisualConfig) -> Tuple[int, int, int]:
    """
    Retorna:
    - capacidad en símbolos 8-CSK,
    - capacidad cruda en bits (3 bits por símbolo),
    - capacidad útil de payload por frame DATA.
    """
    symbol_capacity = len(data_positions(cfg))
    raw_bit_capacity = symbol_capacity * BITS_PER_SYMBOL
    overhead_bits = HEADER_BITS_NO_PAYLOAD + FRAME_CRC_BITS
    payload_capacity_bits = raw_bit_capacity - overhead_bits

    if payload_capacity_bits <= 0:
        raise ValueError(
            "La grilla no tiene capacidad suficiente para el protocolo Fase C. "
            "Aumenta grid_rows/grid_cols o reduce las regiones reservadas."
        )

    return symbol_capacity, raw_bit_capacity, payload_capacity_bits


def print_capacity(cfg: TxVisualConfig) -> None:
    """Imprime la capacidad del formato 8-CSK actual."""
    symbol_capacity, raw_capacity, payload_capacity = get_capacity(cfg)
    print("=== Capacidad visual 8-CSK ===")
    print(f"Celdas DATA: {symbol_capacity} símbolos/frame")
    print(f"Bits por símbolo: {BITS_PER_SYMBOL}")
    print(f"Capacidad cruda: {raw_capacity} bits/frame")
    print(f"Pilotos RGB: {len(pilot_assignments(cfg))} celdas/frame")
    print(f"Referencias distintas: {len(pilot_palette_rgb(cfg))}")
    print(f"Overhead Fase C: {HEADER_BITS_NO_PAYLOAD + FRAME_CRC_BITS} bits/frame")
    print(
        f"Payload útil DATA: {payload_capacity} bits/frame "
        f"= {payload_capacity / 8:.1f} bytes/frame"
    )


def print_csk_constellation(cfg: TxVisualConfig) -> None:
    """Imprime índice, etiqueta Gray, RGB y luminancia digital aproximada."""
    print("=== Constelación 8-CSK ===")
    print("Símbolo | bits | RGB             | Y' aprox.")
    for index, (bits, rgb) in enumerate(zip(GRAY_LABELS, cfg.csk_palette_rgb)):
        r, g, b = rgb
        luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
        bit_text = "".join(map(str, bits))
        print(f"{index:7d} | {bit_text}  | {str(rgb):15s} | {luma:8.2f}")


def make_phase_c_packet_bits(
    payload_bits: np.ndarray,
    frame_type: int,
    sequence: int,
    total_data_frames: int,
    message_len_bytes: int,
    message_crc16: int,
) -> np.ndarray:
    """Crea los bits crudos de un frame Fase C antes del mapeo 8-CSK."""
    if frame_type not in FRAME_TYPE_NAMES:
        raise ValueError(f"frame_type inválido: {frame_type}")

    payload_bits = np.asarray(payload_bits, dtype=np.uint8).reshape(-1)
    if not np.all((payload_bits == 0) | (payload_bits == 1)):
        raise ValueError("payload_bits solo puede contener 0/1.")
    if payload_bits.size >= (1 << 16):
        raise ValueError("payload_bits es demasiado grande para un campo de 16 bits.")

    header_and_payload = np.concatenate([
        PREAMBLE_BITS,
        int_to_bits(PROTOCOL_VERSION, 8),
        int_to_bits(frame_type, 8),
        int_to_bits(sequence, 16),
        int_to_bits(total_data_frames, 16),
        int_to_bits(payload_bits.size, 16),
        int_to_bits(message_len_bytes, 16),
        int_to_bits(message_crc16, 16),
        payload_bits,
    ]).astype(np.uint8)

    frame_crc = crc16_ccitt(bits_to_bytes(header_and_payload))

    return np.concatenate([
        header_and_payload,
        int_to_bits(frame_crc, 16),
    ]).astype(np.uint8)


def grid_offsets(cfg: TxVisualConfig) -> Tuple[int, int]:
    """Offset x,y para centrar la grilla en el frame."""
    grid_w = cfg.grid_cols * cfg.cell_size
    grid_h = cfg.grid_rows * cfg.cell_size

    if grid_w > cfg.frame_width or grid_h > cfg.frame_height:
        raise ValueError(
            "La grilla no cabe en el frame. Reduce cell_size o grid_rows/grid_cols."
        )

    return (cfg.frame_width - grid_w) // 2, (cfg.frame_height - grid_h) // 2


def fill_cell_rgb(
    img_bgr: np.ndarray,
    cfg: TxVisualConfig,
    row: int,
    col: int,
    rgb: Sequence[int],
) -> None:
    """Pinta una celda completa; rgb se proporciona en orden RGB."""
    x_offset, y_offset = grid_offsets(cfg)
    s = cfg.cell_size

    y0 = y_offset + row * s
    y1 = y0 + s
    x0 = x_offset + col * s
    x1 = x0 + s

    img_bgr[y0:y1, x0:x1] = np.asarray(rgb_to_bgr(rgb), dtype=np.uint8)


def draw_finder_marker(
    img_bgr: np.ndarray,
    cfg: TxVisualConfig,
    top_row: int,
    left_col: int,
    marker_type: str = "standard",
) -> None:
    """Dibuja un marcador fiducial 7x7 estándar o ancla."""
    standard_pattern = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ], dtype=np.uint8)

    anchor_pattern = np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
    ], dtype=np.uint8)

    if marker_type == "standard":
        pattern = standard_pattern
    elif marker_type == "anchor":
        pattern = anchor_pattern
    else:
        raise ValueError("marker_type debe ser 'standard' o 'anchor'.")

    for rr in range(7):
        for cc in range(7):
            rgb = cfg.fiducial_high_rgb if pattern[rr, cc] else cfg.fiducial_low_rgb
            fill_cell_rgb(img_bgr, cfg, top_row + rr, left_col + cc, rgb)


def symbols_to_rgb(symbols: np.ndarray, cfg: TxVisualConfig) -> np.ndarray:
    """Mapea índices 0..7 a una matriz Nx3 en orden RGB."""
    symbols = np.asarray(symbols, dtype=np.int16).reshape(-1)
    if np.any(symbols < 0) or np.any(symbols > 7):
        raise ValueError("Los símbolos 8-CSK deben estar entre 0 y 7.")

    palette = np.asarray(cfg.csk_palette_rgb, dtype=np.uint8)
    return palette[symbols]


def render_frame(
    symbols: np.ndarray,
    cfg: TxVisualConfig,
    frame_index: int = 0,
    total_frames: int = 1,
    debug_label: Optional[str] = None,
) -> np.ndarray:
    """
    Renderiza un frame BGR completo con datos 8-CSK, pilotos y fiduciales.

    Las celdas DATA sobrantes se rellenan con una secuencia 0..7 balanceada para
    evitar una región grande dominada por un solo color.
    """
    background_bgr = np.asarray(rgb_to_bgr(cfg.background_rgb), dtype=np.uint8)
    img_bgr = np.empty((cfg.frame_height, cfg.frame_width, 3), dtype=np.uint8)
    img_bgr[:, :] = background_bgr

    roles = build_role_grid(cfg)
    positions = data_positions(cfg)
    symbol_capacity = len(positions)
    symbols = np.asarray(symbols, dtype=np.uint8).reshape(-1)

    if symbols.size > symbol_capacity:
        raise ValueError(
            f"Se recibieron {symbols.size} símbolos, pero la grilla admite "
            f"{symbol_capacity}."
        )
    if np.any(symbols > 7):
        raise ValueError("Los símbolos 8-CSK deben estar entre 0 y 7.")

    filler_len = symbol_capacity - symbols.size
    filler = np.arange(filler_len, dtype=np.uint8) % 8
    full_symbols = np.concatenate([symbols, filler])
    rgb_values = symbols_to_rgb(full_symbols, cfg)

    pilot_lookup = {
        (row, col): ref_index
        for row, col, ref_index in pilot_assignments(cfg)
    }
    pilot_palette = pilot_palette_rgb(cfg)

    data_i = 0
    for r in range(cfg.grid_rows):
        for c in range(cfg.grid_cols):
            role = roles[r, c]

            if role == DATA:
                fill_cell_rgb(img_bgr, cfg, r, c, rgb_values[data_i])
                data_i += 1
            elif role == PILOT:
                ref_index = pilot_lookup[(r, c)]
                fill_cell_rgb(img_bgr, cfg, r, c, pilot_palette[ref_index])
            elif role == QUIET:
                fill_cell_rgb(img_bgr, cfg, r, c, cfg.quiet_rgb)

    # Dibujar fiduciales al final garantiza bordes limpios.
    origins = marker_origins(cfg)
    top_left_origin = min(origins, key=lambda rc: (rc[0], rc[1]))

    for r0, c0 in origins:
        marker_type = "anchor" if (r0, c0) == top_left_origin else "standard"
        draw_finder_marker(img_bgr, cfg, r0, c0, marker_type=marker_type)

    if cfg.debug_text:
        text = debug_label if debug_label else f"Frame {frame_index + 1}/{total_frames}"
        cv2.putText(
            img_bgr,
            text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

    return img_bgr


def _make_record_from_packet(
    packet_bits: np.ndarray,
    frame_type: int,
    sequence: int,
    total_data_frames: int,
    payload_len_bits: int,
    cfg: TxVisualConfig,
    visual_index: int,
    visual_total: int,
) -> TxFrameRecord:
    """Bits de paquete -> símbolos Gray 8-CSK -> imagen + metadatos."""
    symbols, pad_bits = bits_to_csk_symbols(packet_bits)
    symbol_capacity, _, _ = get_capacity(cfg)
    if symbols.size > symbol_capacity:
        raise ValueError(
            f"El paquete requiere {symbols.size} símbolos y solo hay "
            f"{symbol_capacity} celdas DATA."
        )

    label = f"{FRAME_TYPE_NAMES[frame_type]} {sequence}"
    image = render_frame(
        symbols,
        cfg,
        frame_index=visual_index,
        total_frames=visual_total,
        debug_label=label,
    )

    return TxFrameRecord(
        frame_type_name=FRAME_TYPE_NAMES[frame_type],
        frame_type_id=frame_type,
        sequence=sequence,
        total_data_frames=total_data_frames,
        payload_len_bits=payload_len_bits,
        packet_bits=np.asarray(packet_bits, dtype=np.uint8),
        symbols=symbols,
        symbol_padding_bits=pad_bits,
        image=image,
    )


def encode_text_to_records(
    text: str,
    cfg: TxVisualConfig,
    encoding: str = "utf-8",
) -> List[TxFrameRecord]:
    """
    Texto -> registros Fase C completos con modulación 8-CSK.

    Secuencia:
        SYNC... | DATA 0 | DATA 1 | ... | END...
    """
    payload_bytes = text.encode(encoding)
    payload_bits = bytes_to_bits(payload_bytes)
    message_crc = crc16_ccitt(payload_bytes)

    _, _, payload_capacity_bits = get_capacity(cfg)
    total_data_frames = max(1, math.ceil(payload_bits.size / payload_capacity_bits))

    if total_data_frames >= (1 << 16):
        raise ValueError("El mensaje requiere más de 65535 frames DATA.")
    if len(payload_bytes) >= (1 << 16):
        raise ValueError("El mensaje supera 65535 bytes.")

    raw_specs: List[Tuple[int, int, np.ndarray]] = []

    for sync_i in range(cfg.sync_frames):
        raw_specs.append((FRAME_TYPE_SYNC, sync_i, np.array([], dtype=np.uint8)))

    for frame_index in range(total_data_frames):
        start = frame_index * payload_capacity_bits
        end = min(start + payload_capacity_bits, payload_bits.size)
        raw_specs.append((FRAME_TYPE_DATA, frame_index, payload_bits[start:end]))

    for end_i in range(cfg.end_frames):
        raw_specs.append(
            (
                FRAME_TYPE_END,
                total_data_frames + end_i,
                np.array([], dtype=np.uint8),
            )
        )

    records: List[TxFrameRecord] = []
    visual_total = len(raw_specs)

    for visual_index, (frame_type, sequence, chunk) in enumerate(raw_specs):
        packet_bits = make_phase_c_packet_bits(
            payload_bits=chunk,
            frame_type=frame_type,
            sequence=sequence,
            total_data_frames=total_data_frames,
            message_len_bytes=len(payload_bytes),
            message_crc16=message_crc,
        )
        records.append(
            _make_record_from_packet(
                packet_bits=packet_bits,
                frame_type=frame_type,
                sequence=sequence,
                total_data_frames=total_data_frames,
                payload_len_bits=chunk.size,
                cfg=cfg,
                visual_index=visual_index,
                visual_total=visual_total,
            )
        )

    return records


def encode_text_to_frames(
    text: str,
    cfg: TxVisualConfig,
    encoding: str = "utf-8",
) -> List[np.ndarray]:
    """Texto -> lista de imágenes BGR que forman la transmisión Fase C."""
    return [record.image for record in encode_text_to_records(text, cfg, encoding=encoding)]


def get_transmission_summary(
    text: str,
    cfg: TxVisualConfig,
    encoding: str = "utf-8",
) -> dict:
    """Retorna un resumen de capacidad y duración estimada."""
    payload_bytes = text.encode(encoding)
    payload_bits = bytes_to_bits(payload_bytes)
    symbol_capacity, raw_capacity, payload_capacity_bits = get_capacity(cfg)
    total_data_frames = max(1, math.ceil(payload_bits.size / payload_capacity_bits))
    visual_frames = cfg.sync_frames + total_data_frames + cfg.end_frames
    total_time_s = visual_frames * cfg.frame_duration_s

    return {
        "modulation": "8-CSK Gray",
        "protocol_version": PROTOCOL_VERSION,
        "bits_per_symbol": BITS_PER_SYMBOL,
        "message_chars": len(text),
        "message_bytes": len(payload_bytes),
        "data_symbol_capacity": symbol_capacity,
        "raw_capacity_bits": raw_capacity,
        "payload_capacity_bits": payload_capacity_bits,
        "payload_capacity_bytes_equiv": payload_capacity_bits / 8,
        "pilot_cells": len(pilot_assignments(cfg)),
        "pilot_reference_colors": len(pilot_palette_rgb(cfg)),
        "sync_frames": cfg.sync_frames,
        "data_frames": total_data_frames,
        "end_frames": cfg.end_frames,
        "visual_frames_total": visual_frames,
        "frame_duration_s": cfg.frame_duration_s,
        "estimated_tx_time_s": total_time_s,
        "message_crc16": crc16_ccitt(payload_bytes),
    }


def print_transmission_summary(
    text: str,
    cfg: TxVisualConfig,
    encoding: str = "utf-8",
) -> None:
    """Imprime un resumen legible de la transmisión 8-CSK."""
    summary = get_transmission_summary(text, cfg, encoding=encoding)
    print("=== TX Fase C 8-CSK: resumen ===")
    print(f"Modulación: {summary['modulation']}")
    print(f"Versión de protocolo: {summary['protocol_version']}")
    print(f"Texto: {summary['message_chars']} caracteres")
    print(f"Bytes UTF-8: {summary['message_bytes']}")
    print(f"Bits por símbolo: {summary['bits_per_symbol']}")
    print(f"Celdas DATA por frame: {summary['data_symbol_capacity']}")
    print(f"Pilotos RGB por frame: {summary['pilot_cells']}")
    print(f"Frames SYNC: {summary['sync_frames']}")
    print(f"Frames DATA: {summary['data_frames']}")
    print(f"Frames END: {summary['end_frames']}")
    print(f"Frames visuales totales: {summary['visual_frames_total']}")
    print(f"Duración por frame: {summary['frame_duration_s']:.3f} s")
    print(f"Tiempo estimado TX: {summary['estimated_tx_time_s']:.2f} s")
    print(
        f"Payload útil por DATA: {summary['payload_capacity_bits']} bits "
        f"= {summary['payload_capacity_bytes_equiv']:.1f} bytes"
    )
    print(f"CRC16 mensaje: 0x{summary['message_crc16']:04X}")


def save_frames(
    frames: Iterable[np.ndarray],
    cfg: TxVisualConfig,
    output_dir: Optional[Path | str] = None,
    prefix: str = "tx_8csk_frame",
) -> List[Path]:
    """Guarda los frames BGR como PNG y retorna sus rutas."""
    out_dir = Path(output_dir) if output_dir is not None else cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for i, frame in enumerate(frames):
        path = out_dir / f"{prefix}_{i:03d}.png"
        ok = cv2.imwrite(str(path), _to_bgr_uint8(frame))
        if not ok:
            raise IOError(f"No se pudo guardar {path}")
        paths.append(path)

    print(f"Frames guardados en: {out_dir.resolve()}")
    return paths


def _to_bgr_uint8(frame: np.ndarray) -> np.ndarray:
    """Convierte grayscale/BGR/BGRA a BGR uint8 para OpenCV."""
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    raise ValueError(f"Formato de frame no soportado: shape={arr.shape}")


def save_frames_as_video(
    frames: Sequence[np.ndarray],
    cfg: TxVisualConfig,
    output_path: Optional[Path | str] = None,
    video_fps: float = 30.0,
    codec: str = "mp4v",
    repeat_each: Optional[int] = None,
) -> Path:
    """Guarda todos los frames como un video BGR MP4/AVI."""
    if not frames:
        raise ValueError("No hay frames para guardar como video.")
    if video_fps <= 0:
        raise ValueError("video_fps debe ser positivo.")

    out_path = Path(output_path) if output_path is not None else cfg.video_output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    first = _to_bgr_uint8(frames[0])
    height, width = first.shape[:2]

    hold = repeat_each
    if hold is None:
        hold = max(1, int(round(cfg.frame_duration_s * video_fps)))
    if hold <= 0:
        raise ValueError("repeat_each debe ser positivo.")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        str(out_path),
        fourcc,
        float(video_fps),
        (width, height),
        True,
    )

    if not writer.isOpened():
        raise IOError(
            f"No se pudo abrir VideoWriter para {out_path}. "
            "Prueba codec='MJPG' con .avi o codec='mp4v' con .mp4."
        )

    try:
        for frame in frames:
            bgr = _to_bgr_uint8(frame)
            if bgr.shape[:2] != (height, width):
                raise ValueError("Todos los frames deben tener el mismo tamaño.")
            for _ in range(hold):
                writer.write(bgr)
    finally:
        writer.release()

    duration_s = len(frames) * hold / video_fps
    print(
        f"Video guardado en: {out_path.resolve()} | fps={video_fps:g}, "
        f"repeat_each={hold}, duración≈{duration_s:.2f} s"
    )
    return out_path


def show_frame_inline(frame: np.ndarray, title: str = "Frame TX 8-CSK") -> None:
    """Muestra correctamente un frame OpenCV BGR dentro del notebook."""
    if plt is None:
        raise RuntimeError("matplotlib no está disponible en este entorno.")

    bgr = _to_bgr_uint8(frame)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 7))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def preview_frames_inline(
    frames: List[np.ndarray],
    delay_s: float = 0.25,
    max_loops: int = 1,
) -> None:
    """Previsualización sencilla dentro del notebook."""
    if not frames:
        raise ValueError("No hay frames para mostrar.")

    for _ in range(max_loops):
        for i, frame in enumerate(frames):
            if clear_output is not None:
                clear_output(wait=True)
            show_frame_inline(frame, f"Frame 8-CSK {i + 1}/{len(frames)}")
            time.sleep(delay_s)


def transmit_fullscreen_opencv(
    frames: List[np.ndarray],
    cfg: TxVisualConfig,
    fullscreen: bool = True,
    loop: bool = True,
    window_name: str = "TX Fase C 8-CSK",
) -> None:
    """
    Muestra la secuencia con una ventana nativa de OpenCV.

    Controles:
        q o ESC  -> salir
        espacio  -> pausar/reanudar
    """
    if not frames:
        raise ValueError("No hay frames para mostrar.")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if fullscreen:
        cv2.setWindowProperty(
            window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN,
        )
    else:
        cv2.resizeWindow(window_name, cfg.frame_width, cfg.frame_height)

    delay_ms = max(1, int(round(cfg.frame_duration_s * 1000)))
    paused = False

    try:
        while True:
            for frame in frames:
                cv2.imshow(window_name, _to_bgr_uint8(frame))
                key = cv2.waitKey(0 if paused else delay_ms) & 0xFF

                if key in (27, ord("q")):
                    return
                if key == ord(" "):
                    paused = not paused

            if not loop:
                while True:
                    key = cv2.waitKey(100) & 0xFF
                    if key in (27, ord("q")):
                        return
    finally:
        cv2.destroyAllWindows()
        for _ in range(3):
            cv2.waitKey(1)


# La API principal se conserva:
#   cfg = TxVisualConfig()
#   frames = encode_text_to_frames("mensaje", cfg)
#   print_transmission_summary("mensaje", cfg)
#   save_frames_as_video(frames, cfg)
#   transmit_fullscreen_opencv(frames, cfg)
