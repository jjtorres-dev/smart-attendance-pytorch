"""Ejecución integrada de visión, reconocimiento y registro de asistencia."""

from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.attendance.service import (
    AttendancePolicy,
    AttendanceService,
)
from app.database.attendance_repository import (
    AttendanceRepository,
)
from app.recognition.face_detector import (
    FaceDetection,
    YuNetFaceDetector,
)
from app.recognition.face_recognizer import (
    FaceRecognizer,
    RecognitionResult,
)
from app.recognition.track_identity import (
    ConfirmedIdentity,
    associate_faces_to_tracks,
    calculate_track_consensus,
)
from app.vision.crossing import (
    CrossingEvent,
    TrackZoneState,
)
from app.vision.zones import (
    ZONE_LABELS,
    ZoneMap,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class PendingCrossing:
    """Cruce retenido temporalmente hasta confirmar la identidad del track."""

    event: CrossingEvent
    occurred_at: datetime
    track_id: int
    created_at_monotonic: float


def project_path(value: str | Path) -> Path:
    """Resuelve rutas relativas respecto de la raíz del proyecto."""

    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def parse_source(value: str) -> int | str:
    """Interpreta un índice de cámara o conserva una URL o ruta."""

    cleaned_value = value.strip()

    if cleaned_value.isdigit():
        return int(cleaned_value)

    return cleaned_value


def open_capture(
    source: int | str,
) -> cv2.VideoCapture:
    """
    Abre la fuente de video e intenta minimizar la latencia.

    Solicita un tamaño de búfer reducido cuando el backend de OpenCV
    admite esta propiedad y valida que la fuente quede disponible.
    """

    capture = cv2.VideoCapture(source)

    capture.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1,
    )

    if not capture.isOpened():
        capture.release()

        raise RuntimeError(
            f"No se pudo abrir la cámara: {source!r}"
        )

    return capture


def crop_face_with_margin(
    frame: np.ndarray,
    face: FaceDetection,
    margin_ratio: float,
) -> np.ndarray:
    """Recorta una detección facial con margen dentro del fotograma."""

    x, y, width, height = face.box

    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)

    x2 = min(
        frame.shape[1],
        x + width + margin_x,
    )

    y2 = min(
        frame.shape[0],
        y + height + margin_y,
    )

    return frame[y1:y2, x1:x2].copy()


def process_crossing(
    service: AttendanceService,
    pending_crossing: PendingCrossing,
    identity: ConfirmedIdentity,
) -> str:
    """Aplica un cruce identificado al servicio y devuelve su mensaje."""

    if pending_crossing.event == CrossingEvent.ENTRY:
        result = service.handle_entry(
            student_id=identity.student_id,
            occurred_at=pending_crossing.occurred_at,
            track_id=pending_crossing.track_id,
        )
    else:
        result = service.handle_exit(
            student_id=identity.student_id,
            occurred_at=pending_crossing.occurred_at,
            track_id=pending_crossing.track_id,
        )

    status = (
        "ACEPTADO"
        if result.accepted
        else "IGNORADO"
    )

    message = (
        f"{result.occurred_at.strftime('%H:%M:%S')} | "
        f"{identity.student_id} | "
        f"{result.action} | "
        f"{status} | "
        f"{result.message}"
    )

    print(message)

    return message


def draw_person(
    frame: np.ndarray,
    box: np.ndarray,
    track_id: int,
    confidence: float,
    foot_point: tuple[int, int],
    zone_name: str | None,
    identity: ConfirmedIdentity | None,
    latest_result: RecognitionResult | None,
) -> None:
    """Dibuja un track con su zona, confianza e identidad disponible."""

    x1, y1, x2, y2 = box.astype(int)

    if identity is not None:
        color = (0, 255, 0)

        identity_text = (
            f"{identity.student_id} - "
            f"{identity.student_name}"
        )


    elif (

            latest_result is not None

            and latest_result.is_unknown

    ):

        color = (0, 0, 255)

        identity_text = (

            f"DESCONOCIDO | "

            f"Candidato: {latest_result.predicted_class} | "

            f"Conf: {latest_result.confidence:.2f} | "

            f"Margen: {latest_result.margin:.2f}"

        )

    else:
        color = (0, 255, 255)
        identity_text = "SIN IDENTIDAD"

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
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

    first_line = (
        f"Track {track_id} | "
        f"{confidence:.2f} | "
        f"{zone_label}"
    )

    second_line = identity_text

    cv2.putText(
        frame,
        first_line,
        (x1, max(y1 - 38, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        second_line,
        (x1, max(y1 - 12, 52)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_face(
    frame: np.ndarray,
    face: FaceDetection,
    is_associated: bool,
) -> None:
    """Dibuja una detección facial indicando si fue asociada a un track."""

    x, y, width, height = face.box

    color = (
        (255, 255, 0)
        if is_associated
        else (0, 165, 255)
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        color,
        2,
    )

    for landmark in face.landmarks:
        cv2.circle(
            frame,
            landmark,
            2,
            color,
            -1,
        )


def run(args: argparse.Namespace) -> None:
    """Coordina captura, seguimiento, identidad, cruces y persistencia."""

    # Flujo de arquitectura:
    # YOLO detecta personas; ByteTrack mantiene tracks temporales; YuNet
    # detecta el rostro asociado; MobileNetV3 produce una clase candidata
    # y sus probabilidades; confianza y margen aceptan o rechazan cada
    # predicción; el consenso temporal confirma una identidad estable;
    # crossing genera ENTRY o EXIT; AttendanceService aplica las reglas
    # de asistencia; y SQLite conserva los resultados.
    source = parse_source(args.source)

    zones_path = project_path(args.zones)
    attendance_config_path = project_path(
        args.attendance_config
    )
    database_path = project_path(args.database)
    registry_path = project_path(args.registry)

    face_detection_model = project_path(
        args.face_detection_model
    )

    recognition_model = project_path(
        args.recognition_model
    )

    repository = AttendanceRepository(
        database_path
    )

    repository.initialize_schema()

    synchronized_students = (
        repository.sync_students_from_registry(
            registry_path
        )
    )

    policy = AttendancePolicy.from_json(
        attendance_config_path
    )

    service = AttendanceService(
        repository=repository,
        policy=policy,
    )

    timezone = ZoneInfo(
        policy.timezone_name
    )

    zone_map = ZoneMap.load(
        zones_path
    )

    face_detector = YuNetFaceDetector(
        model_path=face_detection_model,
        score_threshold=args.face_confidence,
    )

    face_recognizer = FaceRecognizer(
        checkpoint_path=recognition_model,
        confidence_threshold=(
            args.recognition_threshold
        ),
        margin_threshold=args.margin_threshold,
    )

    if torch.cuda.is_available():
        yolo_device: int | str = 0
        yolo_device_name = torch.cuda.get_device_name(0)
    else:
        yolo_device = "cpu"
        yolo_device_name = "CPU"

    print("Cargando detector de personas...")

    person_model = YOLO(
        args.person_model
    )

    capture = open_capture(source)

    track_zone_states: dict[
        int,
        TrackZoneState,
    ] = {}

    foot_histories: dict[
        int,
        deque[tuple[int, int]],
    ] = defaultdict(
        lambda: deque(maxlen=5)
    )

    recognition_histories: dict[
        int,
        deque[RecognitionResult],
    ] = defaultdict(
        lambda: deque(
            maxlen=args.voting_window
        )
    )

    confirmed_identities: dict[
        int,
        ConfirmedIdentity,
    ] = {}

    latest_recognition_results: dict[
        int,
        RecognitionResult,
    ] = {}

    last_recognition_at: dict[
        int,
        float,
    ] = defaultdict(float)

    last_track_seen_at: dict[
        int,
        float,
    ] = {}

    pending_crossings: dict[
        int,
        PendingCrossing,
    ] = {}

    recent_messages: deque[str] = deque(
        maxlen=7
    )

    closed_session_dates: set[str] = set()

    smoothed_fps = 0.0
    consecutive_failures = 0

    window_name = (
        "Smart Attendance - Sistema integrado"
    )

    print("=" * 78)
    print("SMART ATTENDANCE — SISTEMA INTEGRADO")
    print("=" * 78)
    print(f"Fuente:                  {source}")
    print(f"Zonas:                   {zones_path}")
    print(f"Configuración:           {attendance_config_path}")
    print(f"Base de datos:           {database_path}")
    print(f"Alumnos sincronizados:   {synchronized_students}")
    print(f"Detector de personas:    {args.person_model}")
    print(f"GPU YOLO:                {yolo_device_name}")
    print(f"GPU reconocimiento:      {face_recognizer.device_name}")
    print(f"Entrada límite:          {policy.entry_deadline}")
    print(f"Fin de sesión:           {policy.session_end}")
    print("=" * 78)
    print("Q o ESC para finalizar.")
    print("=" * 78)

    try:
        while True:
            loop_started_at = time.perf_counter()
            now_monotonic = time.monotonic()
            local_now = datetime.now(timezone)

            session_key = local_now.date().isoformat()

            if (
                args.auto_close
                and local_now.time()
                >= policy.session_end
                and session_key
                not in closed_session_dates
            ):
                # El conjunto de fechas cerradas evita repetir el cierre en
                # cada fotograma posterior a la hora final de la sesión.
                summary = service.close_session(
                    local_now.date()
                )

                close_message = (
                    f"{local_now.strftime('%H:%M:%S')} | "
                    f"SESIÓN CERRADA | "
                    f"Total: {summary['TOTAL']} | "
                    f"Puntuales: {summary['ON_TIME']} | "
                    f"Tardanzas: {summary['LATE']} | "
                    f"Ausentes: {summary['ABSENT']}"
                )

                print(close_message)
                recent_messages.append(close_message)

                closed_session_dates.add(
                    session_key
                )

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

            # YOLO detecta exclusivamente personas: la clase 0 de COCO
            # corresponde a "person". Con la configuración predeterminada,
            # ByteTrack usa esas detecciones para mantener IDs temporales;
            # persist=True conserva el estado del seguimiento entre frames.
            tracking_result = person_model.track(
                source=frame,
                persist=True,
                tracker=args.tracker,
                classes=[0],
                conf=args.person_confidence,
                imgsz=args.image_size,
                device=yolo_device,
                verbose=False,
            )[0]

            annotated_frame = frame.copy()

            zone_map.draw(
                annotated_frame
            )

            track_boxes: dict[
                int,
                np.ndarray,
            ] = {}

            track_confidences: dict[
                int,
                float,
            ] = {}

            if (
                tracking_result.boxes is not None
                and tracking_result.boxes.id
                is not None
                and len(tracking_result.boxes) > 0
            ):
                boxes = (
                    tracking_result.boxes.xyxy
                    .detach()
                    .cpu()
                    .numpy()
                )

                track_ids = (
                    tracking_result.boxes.id
                    .detach()
                    .int()
                    .cpu()
                    .tolist()
                )

                confidences = (
                    tracking_result.boxes.conf
                    .detach()
                    .cpu()
                    .tolist()
                )

                for box, track_id, confidence in zip(
                    boxes,
                    track_ids,
                    confidences,
                ):
                    track_boxes[track_id] = box
                    track_confidences[track_id] = (
                        float(confidence)
                    )

                    last_track_seen_at[track_id] = (
                        now_monotonic
                    )

            faces = face_detector.detect(
                frame
            )

            face_associations = (
                associate_faces_to_tracks(
                    faces=faces,
                    track_boxes=track_boxes,
                )
            )

            associated_face_ids = {
                id(face)
                for face in face_associations.values()
            }

            for face in faces:
                draw_face(
                    frame=annotated_frame,
                    face=face,
                    is_associated=(
                        id(face)
                        in associated_face_ids
                    ),
                )

            for track_id, box in track_boxes.items():
                x1, y1, x2, y2 = box.astype(int)

                raw_foot_point = (
                    int((x1 + x2) / 2),
                    int(
                        y1
                        + 0.92
                        * (y2 - y1)
                    ),
                )

                foot_histories[track_id].append(
                    raw_foot_point
                )

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

                face = face_associations.get(
                    track_id
                )

                face_is_large_enough = (
                        face is not None
                        and face.width >= args.minimum_recognition_face
                        and face.height >= args.minimum_recognition_face
                )

                track_is_in_relevant_zone = (
                        detected_zone is not None
                )

                should_recognize = (
                        face_is_large_enough
                        and track_is_in_relevant_zone
                        and (
                                now_monotonic
                                - last_recognition_at[track_id]
                        )
                        >= args.recognition_interval
                )

                # El intervalo limita el costo de inferencia y la zona válida
                # evita acumular identidad cuando la persona está fuera del
                # espacio definido para registrar movimientos.
                if should_recognize:
                    face_crop = crop_face_with_margin(
                        frame=frame,
                        face=face,
                        margin_ratio=args.face_margin,
                    )

                    if (
                        face_crop is not None
                        and face_crop.size > 0
                    ):
                        recognition_result = (
                            face_recognizer.recognize(
                                face_crop
                            )
                        )

                        latest_recognition_results[
                            track_id
                        ] = recognition_result

                        recognition_histories[
                            track_id
                        ].append(
                            recognition_result
                        )

                        last_recognition_at[
                            track_id
                        ] = now_monotonic

                        consensus = (
                            calculate_track_consensus(
                                history=(
                                    recognition_histories[
                                        track_id
                                    ]
                                ),
                                minimum_votes=(
                                    args.minimum_votes
                                ),
                                minimum_vote_ratio=(
                                    args.minimum_vote_ratio
                                ),
                            )
                        )

                        if consensus is not None:
                            confirmed_identities[
                                track_id
                            ] = consensus

                state = track_zone_states.setdefault(
                    track_id,
                    TrackZoneState(),
                )

                previous_zone = (
                    state.confirmed_zone
                )

                crossing_event = state.update(
                    detected_zone=detected_zone,
                    now=now_monotonic,
                    minimum_stable_frames=(
                        args.stable_frames
                    ),
                    event_cooldown_seconds=(
                        args.cooldown
                    ),
                )

                if (
                    state.confirmed_zone
                    != previous_zone
                ):
                    print(
                        f"{local_now.strftime('%H:%M:%S')} | "
                        f"Track {track_id} | "
                        f"ZONA: {previous_zone} "
                        f"→ {state.confirmed_zone}"
                    )

                identity = (
                    confirmed_identities.get(
                        track_id
                    )
                )

                if crossing_event is not None:
                    pending = PendingCrossing(
                        event=crossing_event,
                        occurred_at=local_now,
                        track_id=track_id,
                        created_at_monotonic=(
                            now_monotonic
                        ),
                    )

                    if identity is not None:
                        message = process_crossing(
                            service=service,
                            pending_crossing=pending,
                            identity=identity,
                        )

                        recent_messages.append(
                            message
                        )

                    else:
                        # El cruce se conserva unos segundos: la identidad
                        # puede alcanzar consenso después del evento físico.
                        pending_crossings[
                            track_id
                        ] = pending

                        waiting_message = (
                            f"{local_now.strftime('%H:%M:%S')} | "
                            f"Track {track_id} | "
                            f"{crossing_event.value} | "
                            f"esperando identidad"
                        )

                        print(waiting_message)

                        recent_messages.append(
                            waiting_message
                        )

                draw_person(
                    frame=annotated_frame,
                    box=box,
                    track_id=track_id,
                    confidence=(
                        track_confidences[track_id]
                    ),
                    foot_point=foot_point,
                    zone_name=detected_zone,
                    identity=identity,
                    latest_result=(
                        latest_recognition_results.get(
                            track_id
                        )
                    ),
                )

            for track_id, pending in list(
                pending_crossings.items()
            ):
                identity = (
                    confirmed_identities.get(
                        track_id
                    )
                )

                if identity is not None:
                    message = process_crossing(
                        service=service,
                        pending_crossing=pending,
                        identity=identity,
                    )

                    recent_messages.append(
                        message
                    )

                    del pending_crossings[
                        track_id
                    ]

                    continue

                elapsed_pending = (
                    now_monotonic
                    - pending.created_at_monotonic
                )

                if (
                    elapsed_pending
                    > args.pending_event_timeout
                ):
                    # Un evento sin identidad no se asigna por aproximación;
                    # expira explícitamente para proteger la trazabilidad.
                    expired_message = (
                        f"{local_now.strftime('%H:%M:%S')} | "
                        f"Track {track_id} | "
                        f"{pending.event.value} | "
                        f"NO REGISTRADO: identidad no confirmada"
                    )

                    print(expired_message)

                    recent_messages.append(
                        expired_message
                    )

                    del pending_crossings[
                        track_id
                    ]

            stale_track_ids = [
                track_id
                for track_id, last_seen
                in last_track_seen_at.items()
                if (
                    now_monotonic - last_seen
                    > args.track_timeout
                )
            ]

            for track_id in stale_track_ids:
                # Se eliminan los estados de zona, trayectoria, reconocimiento
                # e identidad del track para evitar herencias si se reutiliza
                # el ID. Los cruces pendientes se gestionan por separado
                # mediante confirmación de identidad o vencimiento.
                track_zone_states.pop(
                    track_id,
                    None,
                )

                foot_histories.pop(
                    track_id,
                    None,
                )

                recognition_histories.pop(
                    track_id,
                    None,
                )

                confirmed_identities.pop(
                    track_id,
                    None,
                )

                latest_recognition_results.pop(
                    track_id,
                    None,
                )

                last_recognition_at.pop(
                    track_id,
                    None,
                )

                last_track_seen_at.pop(
                    track_id,
                    None,
                )

            panel_height = (
                85
                + len(recent_messages) * 25
            )

            cv2.rectangle(
                annotated_frame,
                (0, 0),
                (
                    annotated_frame.shape[1],
                    panel_height,
                ),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                annotated_frame,
                (
                    f"{local_now.strftime('%Y-%m-%d %H:%M:%S')} | "
                    f"Personas: {len(track_boxes)} | "
                    f"Pendientes: {len(pending_crossings)}"
                ),
                (15, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.63,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            elapsed = (
                time.perf_counter()
                - loop_started_at
            )

            instant_fps = (
                1.0 / max(elapsed, 1e-6)
            )

            if smoothed_fps == 0.0:
                smoothed_fps = instant_fps
            else:
                smoothed_fps = (
                    0.90 * smoothed_fps
                    + 0.10 * instant_fps
                )

            cv2.putText(
                annotated_frame,
                (
                    f"FPS: {smoothed_fps:.1f} | "
                    f"Q/ESC: salir"
                ),
                (15, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            for index, message in enumerate(
                reversed(recent_messages)
            ):
                cv2.putText(
                    annotated_frame,
                    message,
                    (15, 92 + index * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            def resize_for_display(
                    frame,
                    max_width: int = 1280,
                    max_height: int = 720,
            ):
                """Reduce solo la visualización, sin alterar el análisis."""

                height, width = frame.shape[:2]

                scale = min(
                    max_width / width,
                    max_height / height,
                    1.0,
                )

                if scale >= 1.0:
                    return frame

                return cv2.resize(
                    frame,
                    (
                        int(width * scale),
                        int(height * scale),
                    ),
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

        print("Sistema finalizado correctamente.")


def build_parser() -> argparse.ArgumentParser:
    """Construye el analizador de configuración del sistema integrado."""

    parser = argparse.ArgumentParser(
        description=(
            "Integra detección, seguimiento, reconocimiento "
            "facial y registro de asistencia."
        )
    )

    parser.add_argument(
        "--minimum-recognition-face",
        type=int,
        default=50,
        help=(
            "Tamaño mínimo en píxeles del rostro "
            "para intentar reconocimiento."
        ),
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
        "--attendance-config",
        default="config/attendance.json",
    )

    parser.add_argument(
        "--database",
        default="data/attendance.db",
    )

    parser.add_argument(
        "--registry",
        default="data/students/students.json",
    )

    parser.add_argument(
        "--person-model",
        default="yolo26n.pt",
    )

    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
    )

    parser.add_argument(
        "--face-detection-model",
        default=(
            "models/face_detection/"
            "face_detection_yunet_2026may.onnx"
        ),
    )

    parser.add_argument(
        "--recognition-model",
        default=(
            "models/recognition/"
            "mobilenet_v3_face_best.pth"
        ),
    )

    parser.add_argument(
        "--person-confidence",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
    )

    parser.add_argument(
        "--face-confidence",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--recognition-threshold",
        type=float,
        default=0.85,
    )

    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--face-margin",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--recognition-interval",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--voting-window",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--minimum-votes",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--minimum-vote-ratio",
        type=float,
        default=0.60,
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
        default=6.0,
    )

    parser.add_argument(
        "--pending-event-timeout",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--auto-close",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser


def main() -> None:
    """Valida los parámetros de consenso e inicia el sistema."""

    parser = build_parser()
    arguments = parser.parse_args()

    if (
        arguments.minimum_votes
        > arguments.voting_window
    ):
        parser.error(
            "--minimum-votes no puede superar "
            "--voting-window."
        )

    if not 0.0 < arguments.minimum_vote_ratio <= 1.0:
        parser.error(
            "--minimum-vote-ratio debe estar entre 0 y 1."
        )

    run(arguments)


if __name__ == "__main__":
    main()
