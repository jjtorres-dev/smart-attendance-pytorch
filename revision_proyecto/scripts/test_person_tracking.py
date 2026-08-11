from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import TypeAlias

import cv2
import numpy as np
import torch
from ultralytics import YOLO


VideoSource: TypeAlias = int | str
Point = tuple[int, int]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_source(value: str) -> VideoSource:
    """
    Convierte valores como '0' en índices de webcam.

    Las URL y rutas de archivos se mantienen como texto.
    """
    cleaned_value = value.strip()

    if cleaned_value.isdigit():
        return int(cleaned_value)

    return cleaned_value


def open_video_source(source: VideoSource) -> cv2.VideoCapture:
    """
    Abre una webcam, cámara IP o archivo de video.
    """
    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"No se pudo abrir la fuente de video: {source!r}"
        )

    return capture


def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    origin: Point,
    font_scale: float = 0.6,
    thickness: int = 2,
) -> None:
    """
    Dibuja texto con fondo para mejorar su visibilidad.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size, baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    x, y = origin
    text_width, text_height = text_size

    cv2.rectangle(
        frame,
        (x, y - text_height - baseline - 8),
        (x + text_width + 8, y + 4),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x + 4, y - 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_person(
    frame: np.ndarray,
    box: np.ndarray,
    track_id: int,
    confidence: float,
    history: deque[Point],
) -> None:
    """
    Dibuja la caja, el ID, la confianza, el punto de los pies
    y la trayectoria reciente de una persona.
    """
    x1, y1, x2, y2 = box.astype(int)

    # El punto inferior central aproxima la posición de los pies.
    foot_x = int((x1 + x2) / 2)
    foot_y = int(y2)
    foot_point = (foot_x, foot_y)

    history.append(foot_point)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    label = f"Persona ID {track_id} | {confidence:.2f}"

    draw_text_with_background(
        frame=frame,
        text=label,
        origin=(x1, max(y1, 30)),
    )

    # Punto que posteriormente usaremos para detectar cruces de zonas.
    cv2.circle(
        frame,
        foot_point,
        6,
        (0, 255, 255),
        -1,
    )

    if len(history) >= 2:
        trajectory = np.array(
            list(history),
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        cv2.polylines(
            frame,
            [trajectory],
            isClosed=False,
            color=(255, 255, 0),
            thickness=2,
        )


def run_tracking(args: argparse.Namespace) -> None:
    source = parse_source(args.source)

    if torch.cuda.is_available():
        device: int | str = 0
        device_name = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        device_name = "CPU"

    print("=" * 65)
    print("DETECCIÓN Y SEGUIMIENTO DE PERSONAS")
    print("=" * 65)
    print(f"Fuente:      {source}")
    print(f"Modelo:      {args.model}")
    print(f"Tracker:     {args.tracker}")
    print(f"Dispositivo: {device_name}")
    print(f"Confianza:   {args.confidence}")
    print(f"Tamaño:      {args.image_size}")
    print("=" * 65)
    print("Controles:")
    print("  Q o ESC = cerrar")
    print("=" * 65)

    print("\nCargando modelo...")
    model = YOLO(args.model)
    print("Modelo cargado.")

    capture = open_video_source(source)

    track_history: dict[int, deque[Point]] = defaultdict(
        lambda: deque(maxlen=args.history_length)
    )

    window_name = "Smart Attendance - Seguimiento de personas"

    smoothed_fps = 0.0
    consecutive_failures = 0

    try:
        while True:
            loop_started_at = time.perf_counter()

            success, frame = capture.read()

            if not success or frame is None:
                consecutive_failures += 1
                print(
                    f"No se recibió el fotograma "
                    f"({consecutive_failures}/30)."
                )

                if consecutive_failures >= 30:
                    raise RuntimeError(
                        "La transmisión dejó de proporcionar video. "
                        "Comprueba IP Webcam y la conexión del hotspot."
                    )

                time.sleep(0.05)
                continue

            consecutive_failures = 0

            # Detecta solamente la clase 0 del conjunto COCO: person.
            result = model.track(
                source=frame,
                persist=True,
                tracker=args.tracker,
                classes=[0],
                conf=args.confidence,
                imgsz=args.image_size,
                device=device,
                verbose=False,
            )[0]

            annotated_frame = frame.copy()
            active_track_ids: set[int] = set()
            person_count = 0

            if (
                result.boxes is not None
                and result.boxes.id is not None
                and len(result.boxes) > 0
            ):
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                track_ids = (
                    result.boxes.id
                    .detach()
                    .int()
                    .cpu()
                    .tolist()
                )
                confidences = (
                    result.boxes.conf
                    .detach()
                    .cpu()
                    .tolist()
                )

                person_count = len(track_ids)

                for box, track_id, confidence in zip(
                    boxes,
                    track_ids,
                    confidences,
                ):
                    active_track_ids.add(track_id)

                    draw_person(
                        frame=annotated_frame,
                        box=box,
                        track_id=track_id,
                        confidence=confidence,
                        history=track_history[track_id],
                    )

            elapsed_time = time.perf_counter() - loop_started_at
            instant_fps = 1.0 / max(elapsed_time, 1e-6)

            if smoothed_fps == 0:
                smoothed_fps = instant_fps
            else:
                smoothed_fps = (
                    0.90 * smoothed_fps
                    + 0.10 * instant_fps
                )

            cv2.putText(
                annotated_frame,
                f"Personas: {person_count}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                annotated_frame,
                f"FPS: {smoothed_fps:.1f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                annotated_frame,
                f"Dispositivo: {device_name}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                annotated_frame,
                "Punto amarillo = referencia para entrada/salida",
                (20, annotated_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

    except KeyboardInterrupt:
        print("\nPrograma interrumpido.")

    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("Cámara y ventanas liberadas correctamente.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detecta y sigue personas desde una webcam, "
            "cámara IP o archivo de video."
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Índice de webcam, URL de cámara IP o ruta de video."
        ),
    )

    parser.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Modelo YOLO que se utilizará.",
    )

    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Configuración del rastreador.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Confianza mínima de detección.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="Tamaño utilizado por el modelo para inferencia.",
    )

    parser.add_argument(
        "--history-length",
        type=int,
        default=30,
        help="Cantidad de puntos conservados por trayectoria.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    run_tracking(arguments)


if __name__ == "__main__":
    main()