from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.recognition.face_detector import YuNetFaceDetector


PROJECT_ROOT = Path(__file__).resolve().parent.parent

STUDENTS_ROOT = PROJECT_ROOT / "data" / "students"
RAW_DATASET_ROOT = STUDENTS_ROOT / "raw"
REGISTRY_PATH = STUDENTS_ROOT / "students.json"

FACE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "face_detection"
    / "face_detection_yunet_2026may.onnx"
)


def parse_source(value: str) -> int | str:
    cleaned_value = value.strip()

    if cleaned_value.isdigit():
        return int(cleaned_value)

    return cleaned_value


def validate_student_id(value: str) -> str:
    cleaned_value = value.strip().upper()

    if not re.fullmatch(r"[A-Z0-9_-]{2,30}", cleaned_value):
        raise argparse.ArgumentTypeError(
            "El código debe tener entre 2 y 30 caracteres y usar "
            "solo letras, números, guiones o guiones bajos."
        )

    return cleaned_value


def open_capture(source: int | str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not capture.isOpened():
        capture.release()

        raise RuntimeError(
            f"No se pudo abrir la fuente de video: {source!r}"
        )

    return capture


def crop_face_with_margin(
    frame: np.ndarray,
    face_box: tuple[int, int, int, int],
    margin_ratio: float,
) -> np.ndarray:
    """
    Recorta el rostro agregando margen alrededor de la detección.
    """

    x, y, width, height = face_box

    margin_x = int(width * margin_ratio)
    margin_y = int(height * margin_ratio)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(frame.shape[1], x + width + margin_x)
    y2 = min(frame.shape[0], y + height + margin_y)

    return frame[y1:y2, x1:x2].copy()


def evaluate_face_quality(
    face_crop: np.ndarray,
    minimum_blur: float,
    minimum_brightness: float,
    maximum_brightness: float,
    minimum_crop_size: int,
) -> tuple[bool, str, float, float]:
    """
    Evalúa nitidez, iluminación y tamaño del rostro.
    """

    if face_crop is None or face_crop.size == 0:
        return False, "Recorte vacío", 0.0, 0.0

    crop_height, crop_width = face_crop.shape[:2]

    if (
        crop_width < minimum_crop_size
        or crop_height < minimum_crop_size
    ):
        return (
            False,
            "Rostro demasiado pequeño",
            0.0,
            0.0,
        )

    gray = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2GRAY,
    )

    blur_score = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    brightness = float(gray.mean())

    if blur_score < minimum_blur:
        return (
            False,
            "Imagen movida o desenfocada",
            blur_score,
            brightness,
        )

    if brightness < minimum_brightness:
        return (
            False,
            "Imagen demasiado oscura",
            blur_score,
            brightness,
        )

    if brightness > maximum_brightness:
        return (
            False,
            "Demasiada iluminación",
            blur_score,
            brightness,
        )

    return (
        True,
        "Calidad correcta",
        blur_score,
        brightness,
    )


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {
            "version": 1,
            "students": [],
        }

    with REGISTRY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        registry = json.load(file)

    if not isinstance(registry.get("students"), list):
        registry["students"] = []

    return registry


def update_registry(
    student_id: str,
    student_name: str,
    student_directory: Path,
    image_count: int,
) -> None:
    STUDENTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry = load_registry()

    student_data = {
        "student_id": student_id,
        "name": student_name,
        "folder": student_directory.relative_to(
            PROJECT_ROOT
        ).as_posix(),
        "image_count": image_count,
        "updated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
    }

    existing_index: int | None = None

    for index, student in enumerate(
        registry["students"]
    ):
        if student.get("student_id") == student_id:
            existing_index = index
            break

    if existing_index is None:
        registry["students"].append(student_data)
    else:
        registry["students"][existing_index] = student_data

    registry["students"].sort(
        key=lambda student: student["student_id"]
    )

    with REGISTRY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            registry,
            file,
            ensure_ascii=False,
            indent=2,
        )


def save_face_image(
    face_crop: np.ndarray,
    student_directory: Path,
    student_id: str,
    image_number: int,
    output_size: int,
) -> Path:
    """
    Redimensiona y guarda un rostro en formato JPG.
    """

    resized_face = cv2.resize(
        face_crop,
        (output_size, output_size),
        interpolation=cv2.INTER_AREA,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    output_path = (
        student_directory
        / (
            f"{student_id}_"
            f"{image_number:03d}_"
            f"{timestamp}.jpg"
        )
    )

    saved = cv2.imwrite(
        str(output_path),
        resized_face,
    )

    if not saved:
        raise RuntimeError(
            f"No se pudo guardar la imagen: {output_path}"
        )

    return output_path


def draw_status_panel(
    frame: np.ndarray,
    student_id: str,
    student_name: str,
    saved_count: int,
    target_count: int,
    automatic_capture: bool,
    quality_message: str,
    face_score: float,
    blur_score: float,
    brightness: float,
) -> None:
    panel_height = 205

    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], panel_height),
        (0, 0, 0),
        -1,
    )

    lines = [
        f"Alumno: {student_id} - {student_name}",
        f"Imagenes guardadas: {saved_count}/{target_count}",
        (
            "Captura automatica: ACTIVADA"
            if automatic_capture
            else "Captura automatica: DESACTIVADA"
        ),
        (
            f"Calidad: {quality_message} | "
            f"YuNet: {face_score:.2f}"
        ),
        (
            f"Nitidez: {blur_score:.1f} | "
            f"Iluminacion: {brightness:.1f}"
        ),
        "A: automatico | ESPACIO: capturar | Q/ESC: salir",
    ]

    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (15, 28 + index * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def run_capture(
    args: argparse.Namespace,
) -> None:
    source = parse_source(args.source)

    student_directory = (
        RAW_DATASET_ROOT / args.student_id
    )

    student_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing_images = sorted(
        student_directory.glob("*.jpg")
    )

    saved_count = len(existing_images)

    if saved_count >= args.target_count:
        print(
            f"El alumno ya tiene {saved_count} imágenes. "
            f"El objetivo es {args.target_count}."
        )

        update_registry(
            student_id=args.student_id,
            student_name=args.name,
            student_directory=student_directory,
            image_count=saved_count,
        )

        return

    detector = YuNetFaceDetector(
        model_path=FACE_MODEL_PATH,
        score_threshold=args.face_confidence,
    )

    capture = open_capture(source)

    automatic_capture = False
    last_saved_at = 0.0

    current_face_crop: np.ndarray | None = None
    current_quality_ok = False

    quality_message = "Buscando rostro"
    face_score = 0.0
    blur_score = 0.0
    brightness = 0.0

    window_name = (
        "Smart Attendance - Registro facial"
    )

    print("=" * 70)
    print("REGISTRO FACIAL DEL ALUMNO")
    print("=" * 70)
    print(f"Código:              {args.student_id}")
    print(f"Nombre:              {args.name}")
    print(f"Fuente:              {source}")
    print(f"Directorio:          {student_directory}")
    print(f"Imágenes existentes: {saved_count}")
    print(f"Objetivo:            {args.target_count}")
    print(f"Confianza YuNet:     {args.face_confidence}")
    print("=" * 70)
    print("A        Activar o desactivar captura automática")
    print("ESPACIO  Capturar imagen manualmente")
    print("Q / ESC  Finalizar")
    print("=" * 70)

    try:
        while True:
            success, frame = capture.read()

            if not success or frame is None:
                print(
                    "Advertencia: no se recibió un fotograma."
                )
                time.sleep(0.05)
                continue

            display_frame = frame.copy()
            faces = detector.detect(frame)

            current_face_crop = None
            current_quality_ok = False

            quality_message = "No se detectó rostro"
            face_score = 0.0
            blur_score = 0.0
            brightness = 0.0

            if len(faces) > 1:
                quality_message = (
                    "Debe aparecer una sola persona"
                )

                for face in faces:
                    x, y, width, height = face.box

                    cv2.rectangle(
                        display_frame,
                        (x, y),
                        (x + width, y + height),
                        (0, 0, 255),
                        2,
                    )

                    cv2.putText(
                        display_frame,
                        f"Rostro {face.score:.2f}",
                        (x, max(y - 8, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

            elif len(faces) == 1:
                face = faces[0]

                face_score = face.score

                x, y, width, height = face.box

                current_face_crop = crop_face_with_margin(
                    frame=frame,
                    face_box=face.box,
                    margin_ratio=args.margin,
                )

                (
                    current_quality_ok,
                    quality_message,
                    blur_score,
                    brightness,
                ) = evaluate_face_quality(
                    face_crop=current_face_crop,
                    minimum_blur=args.minimum_blur,
                    minimum_brightness=(
                        args.minimum_brightness
                    ),
                    maximum_brightness=(
                        args.maximum_brightness
                    ),
                    minimum_crop_size=(
                        args.minimum_crop_size
                    ),
                )

                box_color = (
                    (0, 255, 0)
                    if current_quality_ok
                    else (0, 0, 255)
                )

                cv2.rectangle(
                    display_frame,
                    (x, y),
                    (x + width, y + height),
                    box_color,
                    3,
                )

                cv2.putText(
                    display_frame,
                    f"Rostro {face.score:.2f}",
                    (x, max(y - 8, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    box_color,
                    2,
                    cv2.LINE_AA,
                )

                for landmark in face.landmarks:
                    cv2.circle(
                        display_frame,
                        landmark,
                        3,
                        (0, 255, 255),
                        -1,
                    )

            now = time.monotonic()

            should_capture_automatically = (
                automatic_capture
                and current_quality_ok
                and current_face_crop is not None
                and now - last_saved_at >= args.interval
                and saved_count < args.target_count
            )

            if should_capture_automatically:
                output_path = save_face_image(
                    face_crop=current_face_crop,
                    student_directory=student_directory,
                    student_id=args.student_id,
                    image_number=saved_count + 1,
                    output_size=args.output_size,
                )

                saved_count += 1
                last_saved_at = now

                print(
                    f"Imagen guardada "
                    f"{saved_count}/{args.target_count}: "
                    f"{output_path.name}"
                )

            draw_status_panel(
                frame=display_frame,
                student_id=args.student_id,
                student_name=args.name,
                saved_count=saved_count,
                target_count=args.target_count,
                automatic_capture=automatic_capture,
                quality_message=quality_message,
                face_score=face_score,
                blur_score=blur_score,
                brightness=brightness,
            )

            cv2.imshow(
                window_name,
                display_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if key == ord("a"):
                automatic_capture = (
                    not automatic_capture
                )

                print(
                    "Captura automática:",
                    (
                        "activada"
                        if automatic_capture
                        else "desactivada"
                    ),
                )

            if key == 32:
                if (
                    current_face_crop is not None
                    and current_quality_ok
                    and saved_count < args.target_count
                ):
                    output_path = save_face_image(
                        face_crop=current_face_crop,
                        student_directory=student_directory,
                        student_id=args.student_id,
                        image_number=saved_count + 1,
                        output_size=args.output_size,
                    )

                    saved_count += 1
                    last_saved_at = now

                    print(
                        f"Captura manual guardada "
                        f"{saved_count}/{args.target_count}: "
                        f"{output_path.name}"
                    )
                else:
                    print(
                        "No se guardó la imagen: "
                        f"{quality_message}"
                    )

            if saved_count >= args.target_count:
                print(
                    "Cantidad objetivo alcanzada."
                )
                break

    finally:
        capture.release()
        cv2.destroyAllWindows()

        update_registry(
            student_id=args.student_id,
            student_name=args.name,
            student_directory=student_directory,
            image_count=saved_count,
        )

        print("=" * 70)
        print("REGISTRO FINALIZADO")
        print(f"Alumno:            {args.student_id}")
        print(f"Imágenes guardadas: {saved_count}")
        print(f"Registro:          {REGISTRY_PATH}")
        print("=" * 70)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Captura imágenes faciales de un alumno "
            "utilizando YuNet."
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Índice de webcam, URL o video.",
    )

    parser.add_argument(
        "--student-id",
        required=True,
        type=validate_student_id,
        help="Código único, por ejemplo A001.",
    )

    parser.add_argument(
        "--name",
        required=True,
        help="Nombre completo del alumno.",
    )

    parser.add_argument(
        "--target-count",
        type=int,
        default=60,
        help="Cantidad objetivo de imágenes.",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.40,
        help="Segundos entre capturas automáticas.",
    )

    parser.add_argument(
        "--face-confidence",
        type=float,
        default=0.75,
        help="Confianza mínima de YuNet.",
    )

    parser.add_argument(
        "--minimum-blur",
        type=float,
        default=60.0,
        help="Nitidez mínima requerida.",
    )

    parser.add_argument(
        "--minimum-brightness",
        type=float,
        default=40.0,
    )

    parser.add_argument(
        "--maximum-brightness",
        type=float,
        default=225.0,
    )

    parser.add_argument(
        "--minimum-crop-size",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=0.30,
    )

    parser.add_argument(
        "--output-size",
        type=int,
        default=224,
    )

    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()

    if arguments.target_count < 10:
        parser.error(
            "--target-count debe ser al menos 10."
        )

    if arguments.interval <= 0:
        parser.error(
            "--interval debe ser mayor que cero."
        )

    if not 0.0 < arguments.face_confidence <= 1.0:
        parser.error(
            "--face-confidence debe estar entre 0 y 1."
        )

    run_capture(arguments)


if __name__ == "__main__":
    main()