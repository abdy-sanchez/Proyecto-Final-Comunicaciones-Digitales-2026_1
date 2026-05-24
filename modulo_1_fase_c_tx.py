"""
Módulo TX Fase C: texto -> secuencia multi-frame visual BPSK/Manchester.

Este archivo está pensado para importarse desde un notebook. No ejecuta transmisión
ni abre ventanas al importarlo.

Características principales:
- Mantiene la grilla, marcadores fiduciales, pilotos de brillo y Manchester/BPSK.
- Agrega protocolo multi-frame: SYNC 1 | SYNC 2 | SYNC 3 | DATA... | END.
- Cada frame lleva tipo, número de secuencia, total de frames, longitud de payload,
  longitud del mensaje, CRC global del mensaje y CRC del frame.
- Incluye una función para guardar la secuencia como video MP4/AVI.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

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


@dataclass
class TxVisualConfig:
    # Tamaño del frame generado.
    frame_width: int = 1280
    frame_height: int = 720

    # Grilla visual.
    grid_cols: int = 48
    grid_rows: int = 26
    cell_size: int = 24

    # BPSK visual en escala de grises.
    # Evitamos 0 y 255 para datos, para reducir clipping/saturación en cámara.
    gray_low: int = 64
    gray_high: int = 192

    # Fondo y zonas silenciosas.
    background_gray: int = 16
    quiet_gray: int = 16

    # Marcadores fiduciales tipo finder.
    fiducial_low: int = 0
    fiducial_high: int = 255
    marker_size_cells: int = 7

    # Duración por frame visual durante transmisión real.
    # Para video, se respeta repitiendo cada imagen varias veces según video_fps.
    frame_duration_s: float = 0.12

    # Estructura temporal de la Fase C.
    sync_frames: int = 3
    end_frames: int = 3

    # Carpetas / archivos de salida.
    output_dir: Path = Path("outputs/fase_c/frames")
    video_output_path: Path = Path("outputs/fase_c/tx_sequence.mp4")

    # Útil para depuración, pero mantener False para transmisión real.
    debug_text: bool = False


@dataclass
class TxFrameRecord:
    """Registro interno útil para depurar qué contiene cada imagen TX."""

    frame_type_name: str
    frame_type_id: int
    sequence: int
    total_data_frames: int
    payload_len_bits: int
    packet_bits: np.ndarray
    symbols: np.ndarray
    image: np.ndarray


DATA = "DATA"
FIDUCIAL = "FIDUCIAL"
QUIET = "QUIET"
PILOT_LOW = "PILOT_LOW"
PILOT_HIGH = "PILOT_HIGH"

FRAME_TYPE_SYNC = 1
FRAME_TYPE_DATA = 2
FRAME_TYPE_END = 3

FRAME_TYPE_NAMES = {
    FRAME_TYPE_SYNC: "SYNC",
    FRAME_TYPE_DATA: "DATA",
    FRAME_TYPE_END: "END",
}

PROTOCOL_VERSION = 1

# Preámbulo fijo de 32 bits. El receptor lo puede usar para validar sincronización.
PREAMBLE_BITS = np.array(
    [int(b) for b in "10111000110100101110001001011010"],
    dtype=np.uint8,
)

# Header Fase C, antes del payload:
#   preamble            32 bits
#   protocol_version     8 bits
#   frame_type           8 bits
#   sequence            16 bits
#   total_data_frames   16 bits
#   payload_len_bits    16 bits
#   message_len_bytes   16 bits
#   message_crc16       16 bits
# Después del payload se agrega:
#   frame_crc16         16 bits
HEADER_BITS_NO_PAYLOAD = 32 + 8 + 8 + 16 + 16 + 16 + 16 + 16
FRAME_CRC_BITS = 16


# Configuración por defecto para usar directamente desde notebook si se desea.
cfg = TxVisualConfig()


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Convierte bytes a bits MSB-first."""
    if not data:
        return np.array([], dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Convierte bits MSB-first a bytes. Rellena con ceros si hace falta."""
    bits = np.asarray(bits, dtype=np.uint8)

    if bits.size == 0:
        return b""

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


def manchester_encode(bits: np.ndarray) -> np.ndarray:
    """
    Codificación Manchester.

    Convención:
        bit 0 -> [1, 0]  transición alto-bajo
        bit 1 -> [0, 1]  transición bajo-alto

    Cada bit original produce dos símbolos visuales BPSK.
    """
    bits = np.asarray(bits, dtype=np.uint8)

    if not np.all((bits == 0) | (bits == 1)):
        raise ValueError("Manchester solo acepta bits 0/1.")

    encoded = np.empty(bits.size * 2, dtype=np.uint8)
    encoded[0::2] = 1 - bits
    encoded[1::2] = bits
    return encoded


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


def build_role_grid(cfg: TxVisualConfig) -> np.ndarray:
    """
    Roles de la grilla:
    - DATA: símbolos Manchester/BPSK.
    - FIDUCIAL: marcadores tipo finder.
    - QUIET: zona de guarda alrededor de marcadores.
    - PILOT_HIGH / PILOT_LOW: pilotos de brillo conocidos.
    """
    roles = np.full((cfg.grid_rows, cfg.grid_cols), DATA, dtype=object)
    m = cfg.marker_size_cells

    # Marcadores.
    for r0, c0 in marker_origins(cfg):
        roles[r0:r0 + m, c0:c0 + m] = FIDUCIAL

    # Zona silenciosa de 1 celda alrededor de cada marcador.
    for r0, c0 in marker_origins(cfg):
        r_start = max(0, r0 - 1)
        r_end = min(cfg.grid_rows, r0 + m + 1)
        c_start = max(0, c0 - 1)
        c_end = min(cfg.grid_cols, c0 + m + 1)

        region = roles[r_start:r_end, c_start:c_end]
        region[region != FIDUCIAL] = QUIET

    # Pilotos horizontales.
    top_pilot_row = m + 1
    bottom_pilot_row = cfg.grid_rows - m - 2

    for c in range(m + 1, cfg.grid_cols - m - 1):
        if roles[top_pilot_row, c] == DATA:
            roles[top_pilot_row, c] = PILOT_HIGH if c % 2 == 0 else PILOT_LOW

        if roles[bottom_pilot_row, c] == DATA:
            roles[bottom_pilot_row, c] = PILOT_LOW if c % 2 == 0 else PILOT_HIGH

    # Pilotos verticales.
    left_pilot_col = m + 1
    right_pilot_col = cfg.grid_cols - m - 2

    for r in range(m + 1, cfg.grid_rows - m - 1):
        if roles[r, left_pilot_col] == DATA:
            roles[r, left_pilot_col] = PILOT_HIGH if r % 2 == 0 else PILOT_LOW

        if roles[r, right_pilot_col] == DATA:
            roles[r, right_pilot_col] = PILOT_LOW if r % 2 == 0 else PILOT_HIGH

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
    - capacidad en símbolos visuales Manchester,
    - capacidad en bits crudos antes de Manchester,
    - capacidad útil por frame DATA en bits de payload.
    """
    symbol_capacity = len(data_positions(cfg))
    raw_bit_capacity = symbol_capacity // 2
    overhead_bits = HEADER_BITS_NO_PAYLOAD + FRAME_CRC_BITS
    payload_capacity_bits = raw_bit_capacity - overhead_bits

    if payload_capacity_bits <= 0:
        raise ValueError(
            "La grilla no tiene capacidad suficiente para el protocolo Fase C. "
            "Aumenta grid_rows/grid_cols o reduce marker_size_cells."
        )

    return symbol_capacity, raw_bit_capacity, payload_capacity_bits


def print_capacity(cfg: TxVisualConfig) -> None:
    """Imprime capacidad del formato visual actual."""
    symbol_capacity, raw_capacity, payload_capacity = get_capacity(cfg)
    print(f"Capacidad DATA: {symbol_capacity} símbolos Manchester/frame")
    print(f"Capacidad cruda: {raw_capacity} bits/frame antes de Manchester")
    print(f"Overhead Fase C: {HEADER_BITS_NO_PAYLOAD + FRAME_CRC_BITS} bits/frame")
    print(f"Payload útil DATA: {payload_capacity} bits/frame = {payload_capacity / 8:.1f} bytes/frame")


def make_phase_c_packet_bits(
    payload_bits: np.ndarray,
    frame_type: int,
    sequence: int,
    total_data_frames: int,
    message_len_bytes: int,
    message_crc16: int,
) -> np.ndarray:
    """
    Crea los bits crudos de un frame Fase C antes de Manchester.

    Estructura:
        PREAMBLE_BITS           32 bits
        PROTOCOL_VERSION         8 bits
        frame_type               8 bits   1=SYNC, 2=DATA, 3=END
        sequence                16 bits   DATA: número de frame; SYNC: índice de sync
        total_data_frames       16 bits
        payload_len_bits        16 bits
        message_len_bytes       16 bits
        message_crc16           16 bits   CRC del mensaje completo en bytes
        payload_bits             N bits
        frame_crc16             16 bits   CRC de todo lo anterior
    """
    if frame_type not in FRAME_TYPE_NAMES:
        raise ValueError(f"frame_type inválido: {frame_type}")

    payload_bits = np.asarray(payload_bits, dtype=np.uint8)
    if payload_bits.size >= (1 << 16):
        raise ValueError("payload_bits es demasiado grande para payload_len_bits de 16 bits.")

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
    """Offset para centrar la grilla en el frame."""
    grid_w = cfg.grid_cols * cfg.cell_size
    grid_h = cfg.grid_rows * cfg.cell_size

    if grid_w > cfg.frame_width or grid_h > cfg.frame_height:
        raise ValueError(
            "La grilla no cabe en el frame. "
            "Reduce cell_size o grid_rows/grid_cols."
        )

    return (cfg.frame_width - grid_w) // 2, (cfg.frame_height - grid_h) // 2


def fill_cell(img: np.ndarray, cfg: TxVisualConfig, row: int, col: int, gray: int) -> None:
    """Pinta una celda completa con un nivel de gris."""
    x_offset, y_offset = grid_offsets(cfg)
    s = cfg.cell_size

    y0 = y_offset + row * s
    y1 = y0 + s
    x0 = x_offset + col * s
    x1 = x0 + s

    img[y0:y1, x0:x1] = np.uint8(gray)


def draw_finder_marker(
    img: np.ndarray,
    cfg: TxVisualConfig,
    top_row: int,
    left_col: int,
    marker_type: str = "standard",
) -> None:
    """
    Dibuja un marcador fiducial 7x7.

    marker_type:
        "standard" -> marcador normal usado en 3 esquinas.
        "anchor"   -> marcador especial para identificar la esquina superior izquierda.
    """

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
            gray = cfg.fiducial_high if pattern[rr, cc] else cfg.fiducial_low
            fill_cell(img, cfg, top_row + rr, left_col + cc, gray)


def symbols_to_gray(symbols: np.ndarray, cfg: TxVisualConfig) -> np.ndarray:
    """Mapea símbolos BPSK 0/1 a gray_low/gray_high."""
    symbols = np.asarray(symbols, dtype=np.uint8)

    if not np.all((symbols == 0) | (symbols == 1)):
        raise ValueError("Los símbolos deben ser 0/1.")

    return np.where(symbols == 1, cfg.gray_high, cfg.gray_low).astype(np.uint8)


def render_frame(
    symbols: np.ndarray,
    cfg: TxVisualConfig,
    frame_index: int = 0,
    total_frames: int = 1,
    debug_label: Optional[str] = None,
) -> np.ndarray:
    """Renderiza un frame visual completo con datos, pilotos y fiduciales."""
    img = np.full(
        (cfg.frame_height, cfg.frame_width),
        cfg.background_gray,
        dtype=np.uint8,
    )

    roles = build_role_grid(cfg)
    positions = data_positions(cfg)
    symbol_capacity = len(positions)

    if symbols.size > symbol_capacity:
        raise ValueError(
            f"Se recibieron {symbols.size} símbolos, "
            f"pero la grilla solo admite {symbol_capacity}."
        )

    # Relleno determinístico para celdas DATA no usadas.
    filler_len = symbol_capacity - symbols.size
    filler = np.arange(filler_len, dtype=np.uint8) % 2
    full_symbols = np.concatenate([symbols.astype(np.uint8), filler])
    gray_values = symbols_to_gray(full_symbols, cfg)

    data_i = 0

    for r in range(cfg.grid_rows):
        for c in range(cfg.grid_cols):
            role = roles[r, c]

            if role == DATA:
                fill_cell(img, cfg, r, c, int(gray_values[data_i]))
                data_i += 1
            elif role == PILOT_HIGH:
                fill_cell(img, cfg, r, c, cfg.gray_high)
            elif role == PILOT_LOW:
                fill_cell(img, cfg, r, c, cfg.gray_low)
            elif role == QUIET:
                fill_cell(img, cfg, r, c, cfg.quiet_gray)

    # Fiduciales al final para que queden limpios.
    origins = list(marker_origins(cfg))
    top_left_origin = min(origins, key=lambda rc: (rc[0], rc[1]))

    for r0, c0 in origins:
        if (r0, c0) == top_left_origin:
            draw_finder_marker(img, cfg, r0, c0, marker_type="anchor")
        else:
            draw_finder_marker(img, cfg, r0, c0, marker_type="standard")

    if cfg.debug_text:
        text = debug_label if debug_label else f"Frame {frame_index + 1}/{total_frames}"
        cv2.putText(
            img,
            text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            220,
            2,
            cv2.LINE_AA,
        )

    return img


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
    """Packet crudo -> Manchester -> imagen + metadatos."""
    symbols = manchester_encode(packet_bits)
    label = f"{FRAME_TYPE_NAMES[frame_type]} {sequence}"
    img = render_frame(
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
        packet_bits=packet_bits,
        symbols=symbols,
        image=img,
    )


def encode_text_to_records(
    text: str,
    cfg: TxVisualConfig,
    encoding: str = "utf-8",
) -> List[TxFrameRecord]:
    """
    Texto -> registros Fase C completos.

    Secuencia generada:
        SYNC 1 | SYNC 2 | SYNC 3 | FRAME 0 | FRAME 1 | ... | END | END | END

    Los END se repiten cfg.end_frames veces para aumentar la probabilidad de que el
    receptor capture el delimitador final.
    """
    payload_bytes = text.encode(encoding)
    payload_bits = bytes_to_bits(payload_bytes)
    message_crc = crc16_ccitt(payload_bytes)

    _, _, payload_capacity_bits = get_capacity(cfg)
    total_data_frames = max(1, math.ceil(payload_bits.size / payload_capacity_bits))

    if total_data_frames >= (1 << 16):
        raise ValueError("El mensaje requiere más de 65535 frames DATA.")
    if len(payload_bytes) >= (1 << 16):
        raise ValueError("El mensaje supera 65535 bytes; aumenta el campo message_len_bytes.")

    raw_specs: List[Tuple[int, int, np.ndarray]] = []

    # SYNCs: misma estructura de paquete, payload vacío, secuencia 0..sync_frames-1.
    for sync_i in range(cfg.sync_frames):
        raw_specs.append((FRAME_TYPE_SYNC, sync_i, np.array([], dtype=np.uint8)))

    # DATA frames.
    for frame_index in range(total_data_frames):
        start = frame_index * payload_capacity_bits
        end = min(start + payload_capacity_bits, payload_bits.size)
        chunk = payload_bits[start:end]
        raw_specs.append((FRAME_TYPE_DATA, frame_index, chunk))

    # END repetidos. La secuencia se fija en total_data_frames para que sea distinguible.
    for end_i in range(cfg.end_frames):
        raw_specs.append((FRAME_TYPE_END, total_data_frames + end_i, np.array([], dtype=np.uint8)))

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
        records.append(_make_record_from_packet(
            packet_bits=packet_bits,
            frame_type=frame_type,
            sequence=sequence,
            total_data_frames=total_data_frames,
            payload_len_bits=chunk.size,
            cfg=cfg,
            visual_index=visual_index,
            visual_total=visual_total,
        ))

    return records


def encode_text_to_frames(
    text: str,
    cfg: TxVisualConfig,
    encoding: str = "utf-8",
) -> List[np.ndarray]:
    """Texto -> lista de imágenes/frame Fase C en escala de grises."""
    return [record.image for record in encode_text_to_records(text, cfg, encoding=encoding)]


def get_transmission_summary(text: str, cfg: TxVisualConfig, encoding: str = "utf-8") -> dict:
    """Retorna un resumen simple de la transmisión que se generaría."""
    payload_bytes = text.encode(encoding)
    payload_bits = bytes_to_bits(payload_bytes)
    _, raw_capacity, payload_capacity_bits = get_capacity(cfg)
    total_data_frames = max(1, math.ceil(payload_bits.size / payload_capacity_bits))
    visual_frames = cfg.sync_frames + total_data_frames + cfg.end_frames
    total_time_s = visual_frames * cfg.frame_duration_s

    return {
        "message_chars": len(text),
        "message_bytes": len(payload_bytes),
        "raw_capacity_bits": raw_capacity,
        "payload_capacity_bits": payload_capacity_bits,
        "payload_capacity_bytes_equiv": payload_capacity_bits / 8,
        "sync_frames": cfg.sync_frames,
        "data_frames": total_data_frames,
        "end_frames": cfg.end_frames,
        "visual_frames_total": visual_frames,
        "frame_duration_s": cfg.frame_duration_s,
        "estimated_tx_time_s": total_time_s,
        "message_crc16": crc16_ccitt(payload_bytes),
    }


def print_transmission_summary(text: str, cfg: TxVisualConfig, encoding: str = "utf-8") -> None:
    """Imprime un resumen legible de la transmisión Fase C."""
    summary = get_transmission_summary(text, cfg, encoding=encoding)
    print("=== TX Fase C: resumen ===")
    print(f"Texto: {summary['message_chars']} caracteres")
    print(f"Bytes UTF-8: {summary['message_bytes']}")
    print(f"Frames SYNC: {summary['sync_frames']}")
    print(f"Frames DATA: {summary['data_frames']}")
    print(f"Frames END: {summary['end_frames']}")
    print(f"Frames visuales totales: {summary['visual_frames_total']}")
    print(f"Duración por frame: {summary['frame_duration_s']:.3f} s")
    print(f"Tiempo estimado TX: {summary['estimated_tx_time_s']:.2f} s")
    print(f"Payload útil por DATA: {summary['payload_capacity_bits']} bits = {summary['payload_capacity_bytes_equiv']:.1f} bytes")
    print(f"CRC16 mensaje: 0x{summary['message_crc16']:04X}")


def save_frames(
    frames: Iterable[np.ndarray],
    cfg: TxVisualConfig,
    output_dir: Optional[Path | str] = None,
    prefix: str = "tx_frame",
) -> List[Path]:
    """Guarda los frames como PNG y retorna la lista de rutas."""
    out_dir = Path(output_dir) if output_dir is not None else cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for i, frame in enumerate(frames):
        path = out_dir / f"{prefix}_{i:03d}.png"
        ok = cv2.imwrite(str(path), frame)
        if not ok:
            raise IOError(f"No se pudo guardar {path}")
        paths.append(path)

    print(f"Frames guardados en: {out_dir.resolve()}")
    return paths


def _to_bgr_uint8(frame: np.ndarray) -> np.ndarray:
    """Convierte un frame grayscale/BGR/RGB a BGR uint8 para VideoWriter."""
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
    """
    Guarda todos los frames como un solo video.

    Parámetros:
        frames:
            Lista de imágenes generadas por encode_text_to_frames.
        cfg:
            Configuración visual. Se usa cfg.frame_duration_s si repeat_each=None.
        output_path:
            Ruta de salida. Si None, usa cfg.video_output_path.
        video_fps:
            FPS del archivo de video.
        codec:
            FourCC de OpenCV. Para .mp4 suele funcionar "mp4v".
            Para .avi puedes probar "MJPG".
        repeat_each:
            Número de cuadros de video que se escriben por cada frame TX.
            Si None, se calcula como round(cfg.frame_duration_s * video_fps).

    Retorna:
        Path del video generado.
    """
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
    writer = cv2.VideoWriter(str(out_path), fourcc, float(video_fps), (width, height), True)

    if not writer.isOpened():
        raise IOError(
            f"No se pudo abrir VideoWriter para {out_path}. "
            "Prueba codec='MJPG' y salida .avi, o codec='mp4v' y salida .mp4."
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

    print(
        f"Video guardado en: {out_path.resolve()} | "
        f"fps={video_fps:g}, repeat_each={hold}, duración≈{len(frames) * hold / video_fps:.2f} s"
    )
    return out_path


def show_frame_inline(frame: np.ndarray, title: str = "Frame TX") -> None:
    """Muestra un frame dentro del notebook."""
    if plt is None:
        raise RuntimeError("matplotlib no está disponible en este entorno.")

    plt.figure(figsize=(12, 7))
    plt.imshow(frame, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")
    plt.show()


def preview_frames_inline(frames: List[np.ndarray], delay_s: float = 0.25, max_loops: int = 1) -> None:
    """
    Previsualización sencilla dentro del notebook.
    No reemplaza la transmisión real en pantalla completa.
    """
    if not frames:
        raise ValueError("No hay frames para mostrar.")

    for _ in range(max_loops):
        for i, frame in enumerate(frames):
            if clear_output is not None:
                clear_output(wait=True)
            show_frame_inline(frame, f"Frame {i + 1}/{len(frames)}")
            time.sleep(delay_s)


def transmit_fullscreen_opencv(
    frames: List[np.ndarray],
    cfg: TxVisualConfig,
    fullscreen: bool = True,
    loop: bool = True,
    window_name: str = "TX Fase C BPSK Manchester",
) -> None:
    """
    Muestra los frames usando una ventana nativa de OpenCV.

    Controles:
        q o ESC  -> salir
        espacio  -> pausar / reanudar

    Nota:
    En Jupyter, la ventana OpenCV puede aparecer detrás del navegador.
    Ejecuta esta celda y luego cambia a la ventana que se abrió.
    """
    if not frames:
        raise ValueError("No hay frames para mostrar.")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window_name, cfg.frame_width, cfg.frame_height)

    delay_ms = max(1, int(cfg.frame_duration_s * 1000))
    paused = False

    try:
        while True:
            for frame in frames:
                cv2.imshow(window_name, frame)

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


# Compatibilidad conceptual con el nombre anterior:
# encode_text_to_frames(text, cfg) ahora retorna la secuencia completa de Fase C,
# no solamente los DATA frames.
