from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.vision.crossing import CrossingEvent, TrackZoneState
from app.vision.zones import ZONE_LABELS, ZoneMap


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_source(value: str) -> int | str:
    cleaned_value = value.strip()

    if cleaned_value.isdigit():
        return int(cleaned_value)

    return cleaned_value


def open_capture(source: int | str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la fuente de video: {source!r}"
        )

    return capture


def draw_detection(
    frame: np.ndarray,
    box: np.ndarray,
    track_id: int,
    confidence: float,
    foot_point: tuple[int, int],
    zone_name: str | None,
) -> None:
    x1, y1, x2, y2 = box.astype(int)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2,
    )

    cv2.circle(
        frame,
        foot_point,
        7,
        (255, 0, 255),
        -1,
    )

    zone_label = (
        ZONE_LABELS[zone_name]
        if zone_name is not None
        else "SIN ZONA"
    )

    label = (
        f"ID {track_id} | {confidence:.2f} | {zone_label}"
    )

    label_y = max(y1 - 10, 25)

    cv2.putText(
        frame,
        label,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def run(args: argparse.Namespace) -> None:
    source = parse_source(args.source)

    zones_path = PROJECT_ROOT / args.zones
    zone_map = ZoneMap.load(zones_path)

    if torch.cuda.is_available():
        device: int | str = 0
        device_name = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        device_name = "CPU"

    model = YOLO(args.model)
    capture = open_capture(source)

    track_states: dict[int, TrackZoneState] = {}

    foot_histories: dict[
        int,
        deque[tuple[int, int]],
    ] = defaultdict(lambda: deque(maxlen=5))

    recent_events: deque[tuple[float, str]] = deque(
        maxlen=10
    )

    smoothed_fps = 0.0
    consecutive_failures = 0

    print("=" * 65)
    print("PRUEBA DE ENTRADA Y SALIDA")
    print("=" * 65)
    print(f"Fuente:      {source}")
    print(f"Zonas:       {zones_path}")
    print(f"Modelo:      {args.model}")
    print(f"Dispositivo: {device_name}")
    print("=" * 65)

    window_name = "Smart Attendance - Entrada y salida"

    try:
        while True:
            started_at = time.perf_counter()
            now = time.monotonic()

            success, frame = capture.read()

            if not success or frame is None:
                consecutive_failures += 1

                if consecutive_failures >= 30:
                    raise RuntimeError(
                        "Se perdió la transmisión de la cámara."
                    )

                time.sleep(0.05)
                continue

            consecutive_failures = 0

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
            zone_map.draw(annotated_frame)

            active_ids: set[int] = set()

            if (
                result.boxes is not None
                and result.boxes.id is not None
                and len(result.boxes) > 0
            ):
                boxes = (
                    result.boxes.xyxy
                    .detach()
                    .cpu()
                    .numpy()
                )

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

                for box, track_id, confidence in zip(
                    boxes,
                    track_ids,
                    confidences,
                ):
                    active_ids.add(track_id)

                    x1, y1, x2, y2 = box.astype(int)

                    # Punto inferior ligeramente por encima del final de la caja.
                    raw_foot_point = (
                        int((x1 + x2) / 2),
                        int(y1 + 0.92 * (y2 - y1)),
                    )

                    # Suavizado mediante la mediana de los últimos cinco puntos.
                    foot_histories[track_id].append(raw_foot_point)

                    recent_points = np.asarray(
                        foot_histories[track_id],
                        dtype=np.int32,
                    )

                    median_point = np.median(
                        recent_points,
                        axis=0,
                    ).astype(int)

                    foot_point = (
                        int(median_point[0]),
                        int(median_point[1]),
                    )

                    detected_zone = zone_map.zone_at(
                        point=foot_point,
                        frame_shape=frame.shape,
                    )

                    state = track_states.setdefault(
                        track_id,
                        TrackZoneState(),
                    )

                    previous_confirmed_zone = state.confirmed_zone

                    event = state.update(
                        detected_zone=detected_zone,
                        now=now,
                        minimum_stable_frames=args.stable_frames,
                        event_cooldown_seconds=args.cooldown,
                    )

                    if state.confirmed_zone != previous_confirmed_zone:
                        current_time = time.strftime("%H:%M:%S")

                        print(
                            f"{current_time} | ID {track_id} | "
                            f"ZONA: {previous_confirmed_zone} "
                            f"→ {state.confirmed_zone}"
                        )

                    if event is not None:
                        current_time = time.strftime("%H:%M:%S")

                        if event == CrossingEvent.ENTRY:
                            event_text = (
                                f"{current_time} | "
                                f"ID {track_id} | ENTRADA"
                            )
                        else:
                            event_text = (
                                f"{current_time} | "
                                f"ID {track_id} | SALIDA"
                            )

                        recent_events.append(
                            (now, event_text)
                        )

                        print(event_text)

                    draw_detection(
                        frame=annotated_frame,
                        box=box,
                        track_id=track_id,
                        confidence=confidence,
                        foot_point=foot_point,
                        zone_name=detected_zone,
                    )

            # Eliminar estados antiguos.
            stale_ids = [
                track_id
                for track_id, state in track_states.items()
                if now - state.last_seen_at > args.track_timeout
            ]

            for track_id in stale_ids:
                del track_states[track_id]
                foot_histories.pop(track_id, None)

            # Mantener eventos visibles unos segundos.
            while (
                recent_events
                and now - recent_events[0][0]
                > args.event_display_seconds
            ):
                recent_events.popleft()

            for index, (_, event_text) in enumerate(
                reversed(recent_events)
            ):
                y = 40 + index * 32

                cv2.rectangle(
                    annotated_frame,
                    (10, y - 25),
                    (500, y + 5),
                    (0, 0, 0),
                    -1,
                )

                cv2.putText(
                    annotated_frame,
                    event_text,
                    (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            elapsed = time.perf_counter() - started_at
            instant_fps = 1.0 / max(elapsed, 1e-6)

            if smoothed_fps == 0.0:
                smoothed_fps = instant_fps
            else:
                smoothed_fps = (
                    0.90 * smoothed_fps
                    + 0.10 * instant_fps
                )

            cv2.putText(
                annotated_frame,
                f"FPS: {smoothed_fps:.1f}",
                (annotated_frame.shape[1] - 150, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            def resize_for_display(
                    frame,
                    max_width: int = 1280,
                    max_height: int = 720,
            ):
                """
                Reduce solamente la imagen mostrada en pantalla.

                No modifica el frame utilizado por YOLO,
                ByteTrack ni las zonas.
                """

                height, width = frame.shape[:2]

                scale_width = max_width / width
                scale_height = max_height / height

                scale = min(
                    scale_width,
                    scale_height,
                    1.0,
                )

                if scale >= 1.0:
                    return frame

                new_width = int(width * scale)
                new_height = int(height * scale)

                return cv2.resize(
                    frame,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )

            display_frame = resize_for_display(
                annotated_frame,
                max_width=1280,
                max_height=720,
            )

            cv2.imshow(
                window_name,
                display_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("Programa finalizado correctamente.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prueba cruces entre exterior, puerta e interior."
    )

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--zones",
        default="config/zones.json",
    )

    parser.add_argument(
        "--model",
        default="yolo26n.pt",
    )

    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--stable-frames",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--track-timeout",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--event-display-seconds",
        type=float,
        default=5.0,
    )

    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    run(arguments)


if __name__ == "__main__":
    main()