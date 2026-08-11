"""Prueba en vivo del detector facial YuNet sobre una fuente de video."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from app.recognition.face_detector import YuNetFaceDetector


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "face_detection"
    / "face_detection_yunet_2026may.onnx"
)


def parse_source(value: str) -> int | str:
    """Interpreta un índice de cámara o conserva una URL o ruta."""

    cleaned_value = value.strip()

    if cleaned_value.isdigit():
        return int(cleaned_value)

    return cleaned_value


def run(source: int | str, confidence: float) -> None:
    """Detecta rostros, dibuja referencias y estima el rendimiento."""

    detector = YuNetFaceDetector(
        model_path=MODEL_PATH,
        score_threshold=confidence,
    )

    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"No se pudo abrir la cámara: {source!r}"
        )

    window_name = "Smart Attendance - Deteccion facial YuNet"

    print("=" * 60)
    print("PRUEBA DE DETECCIÓN FACIAL")
    print("=" * 60)
    print(f"Fuente:     {source}")
    print(f"Modelo:     {MODEL_PATH}")
    print(f"Confianza:  {confidence}")
    print("Q o ESC para salir")
    print("=" * 60)

    smoothed_fps = 0.0

    try:
        while True:
            started_at = time.perf_counter()

            success, frame = capture.read()

            if not success or frame is None:
                print("No se recibió un fotograma.")
                time.sleep(0.05)
                continue

            detections = detector.detect(frame)

            for face in detections:
                x, y, width, height = face.box

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Rostro {face.score:.2f}",
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                # YuNet devuelve cinco puntos:
                # ojos, nariz y extremos de la boca.
                for landmark in face.landmarks:
                    cv2.circle(
                        frame,
                        landmark,
                        3,
                        (0, 255, 255),
                        -1,
                    )

            elapsed = time.perf_counter() - started_at
            instant_fps = 1.0 / max(elapsed, 1e-6)

            if smoothed_fps == 0.0:
                smoothed_fps = instant_fps
            else:
                # El promedio exponencial reduce oscilaciones visuales sin
                # ocultar cambios sostenidos en el rendimiento.
                smoothed_fps = (
                    0.90 * smoothed_fps
                    + 0.10 * instant_fps
                )

            cv2.putText(
                frame,
                f"Rostros: {len(detections)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"FPS: {smoothed_fps:.1f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("Prueba finalizada.")


def build_parser() -> argparse.ArgumentParser:
    """Construye el analizador de fuente y confianza mínima."""

    parser = argparse.ArgumentParser(
        description="Prueba el detector facial YuNet."
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Índice de cámara, URL o archivo de video.",
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.75,
        help="Confianza mínima de YuNet.",
    )

    return parser


def main() -> None:
    """Procesa los argumentos e inicia la demostración del detector."""

    parser = build_parser()
    arguments = parser.parse_args()

    run(
        source=parse_source(arguments.source),
        confidence=arguments.confidence,
    )


if __name__ == "__main__":
    main()
