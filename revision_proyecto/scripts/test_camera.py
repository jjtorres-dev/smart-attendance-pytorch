from __future__ import annotations

import argparse
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import TypeAlias

import cv2


CameraSource: TypeAlias = int | str

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "test_videos"


def parse_source(value: str) -> CameraSource:
    """
    Convierte valores numéricos como '0' o '1' en índices de cámara.

    Cualquier otro valor se considera una URL o ruta de video.
    """
    stripped_value = value.strip()

    if stripped_value.isdigit():
        return int(stripped_value)

    return stripped_value


def open_capture(
    source: CameraSource,
    width: int,
    height: int,
    requested_fps: int,
) -> cv2.VideoCapture:
    """
    Abre una cámara física, una URL o un archivo de video.

    En Windows intenta DirectShow para cámaras locales y posteriormente
    utiliza el backend automático como alternativa.
    """
    capture: cv2.VideoCapture | None = None

    is_local_camera = isinstance(source, int)
    is_windows = platform.system() == "Windows"

    if is_local_camera and is_windows:
        print("Intentando abrir la cámara mediante DirectShow...")
        capture = cv2.VideoCapture(source, cv2.CAP_DSHOW)

    if capture is None or not capture.isOpened():
        if capture is not None:
            capture.release()

        print("Intentando abrir mediante el backend automático...")
        capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"No se pudo abrir la fuente de video: {source!r}.\n"
            "Comprueba el índice de cámara, la URL y los permisos de Windows."
        )

    # Estas propiedades funcionan principalmente con cámaras locales.
    if is_local_camera:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, requested_fps)

    # Intenta reducir el retraso en transmisiones de red.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return capture


def print_capture_information(
    capture: cv2.VideoCapture,
    source: CameraSource,
) -> None:
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = capture.get(cv2.CAP_PROP_FPS)
    backend = capture.getBackendName()

    print("=" * 60)
    print("CÁMARA ABIERTA")
    print("=" * 60)
    print(f"Fuente:       {source}")
    print(f"Backend:      {backend}")
    print(f"Resolución:   {actual_width} x {actual_height}")
    print(f"FPS indicado: {actual_fps:.2f}")
    print()
    print("Controles:")
    print("  Q o ESC = salir")
    print("  R       = iniciar o detener grabación")
    print("=" * 60)


def create_video_writer(
    frame,
    requested_fps: int,
) -> tuple[cv2.VideoWriter, Path]:
    VIDEO_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = VIDEO_OUTPUT_DIRECTORY / f"camera_test_{timestamp}.mp4"

    frame_height, frame_width = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        float(requested_fps),
        (frame_width, frame_height),
    )

    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            "OpenCV no pudo crear el archivo de video. "
            "Comprueba que la carpeta data/test_videos sea accesible."
        )

    return writer, output_path


def draw_information(
    frame,
    measured_fps: float,
    recording: bool,
) -> None:
    status_text = "GRABANDO" if recording else "VISTA PREVIA"

    cv2.putText(
        frame,
        f"FPS: {measured_fps:.1f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        status_text,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if recording else (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Q/ESC: salir | R: grabar",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def run_camera_test(args: argparse.Namespace) -> None:
    source = parse_source(args.source)

    capture = open_capture(
        source=source,
        width=args.width,
        height=args.height,
        requested_fps=args.fps,
    )

    print_capture_information(capture, source)

    writer: cv2.VideoWriter | None = None
    recording = False
    output_path: Path | None = None

    fps_started_at = time.perf_counter()
    frames_for_fps = 0
    measured_fps = 0.0
    consecutive_failures = 0

    window_name = "Smart Attendance - Prueba de camara"

    try:
        while True:
            success, frame = capture.read()

            if not success or frame is None:
                consecutive_failures += 1

                if consecutive_failures == 1:
                    print("Advertencia: no se recibió un fotograma.")

                if consecutive_failures >= 30:
                    print("Reconectando la fuente de video...")

                    capture.release()
                    time.sleep(1)

                    capture = open_capture(
                        source=source,
                        width=args.width,
                        height=args.height,
                        requested_fps=args.fps,
                    )

                    consecutive_failures = 0

                continue

            consecutive_failures = 0

            frames_for_fps += 1
            elapsed_time = time.perf_counter() - fps_started_at

            if elapsed_time >= 1:
                measured_fps = frames_for_fps / elapsed_time
                frames_for_fps = 0
                fps_started_at = time.perf_counter()

            if recording and writer is not None:
                writer.write(frame)

            draw_information(
                frame=frame,
                measured_fps=measured_fps,
                recording=recording,
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("r"):
                if not recording:
                    writer, output_path = create_video_writer(
                        frame=frame,
                        requested_fps=args.fps,
                    )

                    recording = True
                    print(f"Grabación iniciada: {output_path}")
                else:
                    recording = False

                    if writer is not None:
                        writer.release()
                        writer = None

                    print(f"Grabación finalizada: {output_path}")

    except KeyboardInterrupt:
        print("\nEjecución interrumpida por el usuario.")

    finally:
        if writer is not None:
            writer.release()

        capture.release()
        cv2.destroyAllWindows()

        print("Cámara liberada correctamente.")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prueba una webcam, cámara IP o archivo de video."
    )

    parser.add_argument(
        "--source",
        default="0",
        help=(
            "Índice de webcam, URL de cámara IP o ruta de archivo. "
            "Ejemplos: 0, 1, http://192.168.1.50:8080/video"
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Ancho solicitado para una cámara local.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Alto solicitado para una cámara local.",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="FPS solicitados y utilizados para grabar.",
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    arguments = parser.parse_args()
    run_camera_test(arguments)


if __name__ == "__main__":
    main()