"""
video_recorder_module.py

Modulo auxiliar para grabar video crudo desde la camara sin procesar imagenes.
Pensado para usar desde Jupyter durante la Fase C del modem optico.

Uso tipico:

    import video_recorder_module as vr

    info = vr.record_video_on_key(
        duration_s=10,
        camera_index=0,
        width=1280,
        height=720,
        output_dir="outputs/fase_c/raw_recordings",
    )

    print(info["output_path"])

Controles:
    ESPACIO / ENTER / s / r  -> iniciar grabacion
    q / ESC                  -> cancelar antes de grabar o detener durante grabacion

Nota:
    El video se guarda tal como entrega los frames la camara. No hay deteccion,
    rectificacion, demodulacion ni procesamiento de imagenes. Solo se muestra una
    vista previa en una ventana de OpenCV.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2


@dataclass
class RawVideoRecorderConfig:
    """Configuracion basica del grabador de video crudo."""

    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    duration_s: float = 10.0
    output_dir: Union[str, Path] = Path("outputs/fase_c/raw_recordings")
    output_filename: Optional[str] = None
    codec: str = "mp4v"
    window_name: str = "Grabador RX - video crudo"
    warmup_frames: int = 10
    show_preview: bool = True
    autofocus_off: bool = False


def _make_output_path(
    output_dir: Union[str, Path],
    output_filename: Optional[str],
    extension: str = ".mp4",
) -> Path:
    """Construye la ruta de salida y crea la carpeta si no existe."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_filename is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"raw_camera_recording_{stamp}{extension}"

    path = Path(output_filename)
    if path.suffix == "":
        path = path.with_suffix(extension)

    if not path.is_absolute():
        path = out_dir / path

    return path


def _safe_destroy_windows() -> None:
    """Cierra ventanas OpenCV de forma robusta, util en Jupyter."""
    try:
        cv2.destroyAllWindows()
        for _ in range(3):
            cv2.waitKey(1)
    except cv2.error:
        pass


def record_video_on_key(
    duration_s: float = 10.0,
    camera_index: int = 0,
    width: int = 1280,
    height: int = 720,
    output_dir: Union[str, Path] = Path("outputs/fase_c/raw_recordings"),
    output_filename: Optional[str] = None,
    fps: float = 30.0,
    codec: str = "mp4v",
    window_name: str = "Grabador RX - video crudo",
    warmup_frames: int = 10,
    show_preview: bool = True,
    autofocus_off: bool = False,
) -> Dict[str, Any]:
    """
    Enciende la camara, espera una tecla para iniciar y graba video crudo.

    Parametros principales:
        duration_s:
            Duracion de grabacion en segundos.
        camera_index:
            Indice de camara para cv2.VideoCapture. Usualmente 0 es la camara principal.
        width, height:
            Resolucion solicitada a la camara. La camara puede entregar otra resolucion.
        output_dir:
            Carpeta donde se guarda el video.
        output_filename:
            Nombre opcional del archivo. Si se omite, se genera con timestamp.
        fps:
            FPS usado por el VideoWriter. No siempre coincide exactamente con el FPS real de captura.
        codec:
            Codigo FourCC. Para .mp4 suele funcionar "mp4v". Para .avi puedes probar "XVID".
        show_preview:
            Si True, muestra una ventana OpenCV durante espera y grabacion.
        autofocus_off:
            Si True, intenta desactivar autofocus. No todas las camaras lo soportan.

    Retorna:
        Diccionario con ruta de salida, cantidad de frames y FPS efectivo aproximado.
    """
    if duration_s <= 0:
        raise ValueError("duration_s debe ser mayor que cero.")
    if fps <= 0:
        raise ValueError("fps debe ser mayor que cero.")
    if len(codec) != 4:
        raise ValueError("codec debe tener exactamente 4 caracteres, por ejemplo 'mp4v' o 'XVID'.")

    output_path = _make_output_path(output_dir, output_filename)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la camara con indice {camera_index}. "
            "Prueba con camera_index=1 o revisa permisos de camara."
        )

    writer = None
    recording_started = False
    frame_count = 0
    start_time = None
    end_time = None

    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_FPS, float(fps))

        if autofocus_off:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

        for _ in range(max(0, int(warmup_frames))):
            cap.read()
            time.sleep(0.01)

        if show_preview:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # 1) Espera de inicio
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("La camara no entrego frame durante la vista previa.")

            if show_preview:
                preview = frame.copy()
                cv2.putText(
                    preview,
                    "ESPACIO/ENTER/s/r: grabar | q/ESC: cancelar",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = ord("r")

            if key in (13, 32, ord("s"), ord("r")):
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (w, h))

                if not writer.isOpened():
                    raise RuntimeError(
                        f"No se pudo crear el archivo de video: {output_path}. "
                        "Prueba otro codec, por ejemplo codec='XVID' y extension .avi."
                    )

                recording_started = True
                start_time = time.monotonic()
                break

            if key in (27, ord("q")):
                return {
                    "ok": False,
                    "cancelled": True,
                    "recording_started": False,
                    "output_path": None,
                    "frames_recorded": 0,
                    "duration_s_requested": float(duration_s),
                    "duration_s_actual": 0.0,
                    "effective_fps": 0.0,
                    "message": "Grabacion cancelada antes de iniciar.",
                }

        # 2) Grabacion cruda durante duration_s
        while True:
            now = time.monotonic()
            elapsed = now - start_time
            if elapsed >= duration_s:
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("La camara dejo de entregar frames durante la grabacion.")

            # Guardado crudo: no se rectifica, no se filtra, no se decodifica.
            writer.write(frame)
            frame_count += 1

            if show_preview:
                preview = frame.copy()
                remaining = max(0.0, duration_s - elapsed)
                cv2.putText(
                    preview,
                    f"REC {elapsed:05.2f}s / {duration_s:.2f}s | faltan {remaining:05.2f}s | q/ESC: detener",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(window_name, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

        end_time = time.monotonic()

    finally:
        if writer is not None:
            writer.release()
        cap.release()
        if show_preview:
            _safe_destroy_windows()

    actual_duration = 0.0 if start_time is None or end_time is None else end_time - start_time
    effective_fps = frame_count / actual_duration if actual_duration > 0 else 0.0

    return {
        "ok": bool(recording_started and frame_count > 0 and output_path.exists()),
        "cancelled": False,
        "recording_started": bool(recording_started),
        "output_path": str(output_path.resolve()) if output_path.exists() else str(output_path),
        "frames_recorded": int(frame_count),
        "duration_s_requested": float(duration_s),
        "duration_s_actual": float(actual_duration),
        "effective_fps": float(effective_fps),
        "codec": codec,
        "fps_writer": float(fps),
        "camera_index": int(camera_index),
        "message": "Video guardado correctamente." if frame_count > 0 else "No se grabaron frames.",
    }


def record_video_with_config(config: RawVideoRecorderConfig) -> Dict[str, Any]:
    """Version conveniente usando dataclass de configuracion."""
    return record_video_on_key(
        duration_s=config.duration_s,
        camera_index=config.camera_index,
        width=config.width,
        height=config.height,
        output_dir=config.output_dir,
        output_filename=config.output_filename,
        fps=config.fps,
        codec=config.codec,
        window_name=config.window_name,
        warmup_frames=config.warmup_frames,
        show_preview=config.show_preview,
        autofocus_off=config.autofocus_off,
    )


__all__ = [
    "RawVideoRecorderConfig",
    "record_video_on_key",
    "record_video_with_config",
]
