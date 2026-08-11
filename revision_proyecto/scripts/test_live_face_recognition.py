from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

from app.recognition.face_detector import (
    FaceDetection,
    YuNetFaceDetector,
)
from app.recognition.face_recognizer import (
    FaceRecognizer,
    RecognitionResult,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

FACE_DETECTION_MODEL = (
    PROJECT_ROOT
    / "models"
    / "face_detection"
    / "face_detection_yunet_2026may.onnx"
)

RECOGNITION_MODEL = (
    PROJECT_ROOT
    / "models"
    / "recognition"
    / "mobilenet_v3_face_best.pth"
)


def parse_source(value: str) -> int | str:
    cleaned_value = value.strip()

    if cleaned_value.isdigit():
        return int(cleaned_value)

    return cleaned_value


def crop_face_with_margin(
    frame: np.ndarray,
    face: FaceDetection,
    margin_ratio: float,
) -> np.ndarray:
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


def calculate_consensus(
    history: deque[RecognitionResult],
    minimum_votes: int,
) -> tuple[str, str, float] | None:
    """
    Confirma una identidad usando varias predicciones consecutivas.
    """

    known_results = [
        result
        for result in history
        if (
            not result.is_unknown
            and result.student_id is not None
        )
    ]

    if len(known_results) < minimum_votes:
        return None

    vote_counter = Counter(
        result.student_id
        for result in known_results
    )

    student_id, votes = (
        vote_counter.most_common(1)[0]
    )

    if votes < minimum_votes:
        return None

    matching_results = [
        result
        for result in known_results
        if result.student_id == student_id
    ]

    average_confidence = sum(
        result.confidence
        for result in matching_results
    ) / len(matching_results)

    student_name = matching_results[-1].student_name

    return (
        student_id,
        student_name,
        average_confidence,
    )


def draw_face_result(
    frame: np.ndarray,
    face: FaceDetection,
    result: RecognitionResult,
) -> None:
    x, y, width, height = face.box

    color = (
        (0, 0, 255)
        if result.is_unknown
        else (0, 255, 0)
    )

    cv2.rectangle(
        frame,
        (x, y),
        (x + width, y + height),
        color,
        3,
    )

    if result.is_unknown:
        identity_text = "DESCONOCIDO"
    else:
        identity_text = (
            f"{result.student_id} - "
            f"{result.student_name}"
        )

    cv2.putText(
        frame,
        identity_text,
        (x, max(y - 42, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        (
            f"Conf: {result.confidence:.2f} | "
            f"Margen: {result.margin:.2f}"
        ),
        (x, max(y - 15, 52)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )

    for landmark in face.landmarks:
        cv2.circle(
            frame,
            landmark,
            3,
            (0, 255, 255),
            -1,
        )


def run(args: argparse.Namespace) -> None:
    source = parse_source(args.source)

    detector = YuNetFaceDetector(
        model_path=FACE_DETECTION_MODEL,
        score_threshold=args.face_confidence,
    )

    recognizer = FaceRecognizer(
        checkpoint_path=RECOGNITION_MODEL,
        confidence_threshold=(
            args.recognition_threshold
        ),
        margin_threshold=args.margin_threshold,
    )

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

    recognition_history: deque[
        RecognitionResult
    ] = deque(maxlen=args.voting_window)

    last_face_seen_at = 0.0
    smoothed_fps = 0.0

    print("=" * 72)
    print("RECONOCIMIENTO FACIAL EN VIVO")
    print("=" * 72)
    print(f"Fuente:                 {source}")
    print(f"Clases:                 {recognizer.classes}")
    print(f"Dispositivo:            {recognizer.device_name}")
    print(
        f"Umbral reconocimiento:  "
        f"{args.recognition_threshold}"
    )
    print(
        f"Umbral margen:          "
        f"{args.margin_threshold}"
    )
    print(
        f"Ventana de votación:    "
        f"{args.voting_window}"
    )
    print(
        f"Votos mínimos:          "
        f"{args.minimum_votes}"
    )
    print("=" * 72)
    print("Durante esta prueba debe aparecer una persona a la vez.")
    print("Q o ESC para salir.")
    print("=" * 72)

    window_name = (
        "Smart Attendance - Reconocimiento facial"
    )

    try:
        while True:
            started_at = time.perf_counter()
            now = time.monotonic()

            success, frame = capture.read()

            if not success or frame is None:
                print(
                    "No se recibió un fotograma."
                )
                time.sleep(0.05)
                continue

            faces = detector.detect(frame)

            current_result: (
                RecognitionResult | None
            ) = None

            if len(faces) >= 1:
                # YuNet devuelve primero el rostro de mayor área.
                primary_face = faces[0]

                face_crop = crop_face_with_margin(
                    frame=frame,
                    face=primary_face,
                    margin_ratio=args.margin,
                )

                current_result = recognizer.recognize(
                    face_crop
                )

                recognition_history.append(
                    current_result
                )

                last_face_seen_at = now

                draw_face_result(
                    frame=frame,
                    face=primary_face,
                    result=current_result,
                )

                # En esta fase solo reconocemos el rostro principal.
                for extra_face in faces[1:]:
                    x, y, width, height = extra_face.box

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + width, y + height),
                        (0, 255, 255),
                        2,
                    )

                    cv2.putText(
                        frame,
                        "ROSTRO ADICIONAL",
                        (x, max(y - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            elif now - last_face_seen_at > 1.5:
                recognition_history.clear()

            consensus = calculate_consensus(
                history=recognition_history,
                minimum_votes=args.minimum_votes,
            )

            panel_color = (0, 0, 0)

            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], 135),
                panel_color,
                -1,
            )

            if consensus is None:
                confirmed_text = (
                    "IDENTIDAD: SIN CONFIRMAR"
                )
                confirmed_color = (0, 255, 255)
            else:
                (
                    confirmed_id,
                    confirmed_name,
                    average_confidence,
                ) = consensus

                confirmed_text = (
                    f"IDENTIDAD CONFIRMADA: "
                    f"{confirmed_id} - {confirmed_name} "
                    f"({average_confidence:.2f})"
                )

                confirmed_color = (0, 255, 0)

            cv2.putText(
                frame,
                confirmed_text,
                (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                confirmed_color,
                2,
                cv2.LINE_AA,
            )

            raw_text = "Prediccion actual: sin rostro"

            if current_result is not None:
                raw_name = (
                    "DESCONOCIDO"
                    if current_result.is_unknown
                    else (
                        f"{current_result.student_id} - "
                        f"{current_result.student_name}"
                    )
                )

                raw_text = (
                    f"Prediccion actual: {raw_name} | "
                    f"{current_result.confidence:.2f}"
                )

            cv2.putText(
                frame,
                raw_text,
                (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            elapsed = (
                time.perf_counter()
                - started_at
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
                frame,
                (
                    f"Rostros: {len(faces)} | "
                    f"FPS: {smoothed_fps:.1f}"
                ),
                (15, 112),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                window_name,
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()

        print(
            "Reconocimiento finalizado."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prueba el reconocimiento facial "
            "en video en tiempo real."
        )
    )

    parser.add_argument(
        "--source",
        required=True,
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
        "--margin",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--voting-window",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--minimum-votes",
        type=int,
        default=8,
    )

    return parser


def main() -> None:
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

    run(arguments)


if __name__ == "__main__":
    main()