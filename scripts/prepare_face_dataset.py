"""Preparación reproducible del conjunto facial de entrenamiento y evaluación."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Final


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_RAW_ROOT = PROJECT_ROOT / "data" / "students" / "raw"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "students" / "dataset"
DEFAULT_REGISTRY_PATH = (
    PROJECT_ROOT / "data" / "students" / "students.json"
)

IMAGE_EXTENSIONS: Final[set[str]] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def find_images(directory: Path) -> list[Path]:
    """Devuelve las imágenes válidas de una carpeta."""

    return sorted(
        file_path
        for file_path in directory.iterdir()
        if (
            file_path.is_file()
            and file_path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )


def load_student_registry(path: Path) -> dict[str, str]:
    """
    Devuelve un diccionario:
    código del alumno -> nombre completo.
    """

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    students = data.get("students", [])

    return {
        str(student["student_id"]): str(student["name"])
        for student in students
        if "student_id" in student and "name" in student
    }


def calculate_split_sizes(
    total_images: int,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[int, int, int]:
    """
    Calcula cantidades garantizando al menos una imagen
    en entrenamiento, validación y prueba.
    """

    if total_images < 3:
        raise ValueError(
            "Se necesitan al menos tres imágenes por alumno."
        )

    train_count = max(
        1,
        int(total_images * train_ratio),
    )

    validation_count = max(
        1,
        int(total_images * validation_ratio),
    )

    test_count = (
        total_images
        - train_count
        - validation_count
    )

    while test_count < 1:
        if train_count > validation_count and train_count > 1:
            train_count -= 1

        elif validation_count > 1:
            validation_count -= 1

        else:
            raise ValueError(
                "No se pudo crear una división válida."
            )

        test_count = (
            total_images
            - train_count
            - validation_count
        )

    return (
        train_count,
        validation_count,
        test_count,
    )


def copy_images(
    images: list[Path],
    destination: Path,
) -> None:
    """Copia imágenes a una partición conservando metadatos del archivo."""

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    for image_path in images:
        destination_path = (
            destination / image_path.name
        )

        shutil.copy2(
            image_path,
            destination_path,
        )


def validate_arguments(
    train_ratio: float,
    validation_ratio: float,
) -> None:
    """Valida que las proporciones de partición sean positivas y compatibles."""

    if not 0.0 < train_ratio < 1.0:
        raise ValueError(
            "--train-ratio debe estar entre 0 y 1."
        )

    if not 0.0 < validation_ratio < 1.0:
        raise ValueError(
            "--validation-ratio debe estar entre 0 y 1."
        )

    if train_ratio + validation_ratio >= 1.0:
        raise ValueError(
            "La suma de entrenamiento y validación "
            "debe ser menor que 1."
        )


def prepare_dataset(
    raw_root: Path,
    output_root: Path,
    registry_path: Path,
    train_ratio: float,
    validation_ratio: float,
    minimum_images: int,
    seed: int,
    clean_output: bool,
) -> None:
    """Divide las imágenes por estudiante y genera un manifiesto descriptivo."""

    validate_arguments(
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )

    if not raw_root.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset original: {raw_root}"
        )

    student_directories = sorted(
        directory
        for directory in raw_root.iterdir()
        if directory.is_dir()
    )

    if len(student_directories) < 2:
        raise RuntimeError(
            "Se necesitan al menos dos alumnos registrados."
        )

    if clean_output and output_root.exists():
        print(
            f"Eliminando dataset anterior: {output_root}"
        )

        shutil.rmtree(output_root)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    student_names = load_student_registry(
        registry_path
    )

    manifest_students: list[dict] = []
    total_train = 0
    total_validation = 0
    total_test = 0

    print("=" * 72)
    print("PREPARACIÓN DEL DATASET FACIAL")
    print("=" * 72)
    print(f"Origen:       {raw_root}")
    print(f"Destino:      {output_root}")
    print(f"Semilla:      {seed}")
    print(f"Entrenamiento:{train_ratio:.0%}")
    print(f"Validación:   {validation_ratio:.0%}")
    print(
        f"Prueba:       "
        f"{1.0 - train_ratio - validation_ratio:.0%}"
    )
    print("=" * 72)

    for class_index, student_directory in enumerate(
        student_directories
    ):
        student_id = student_directory.name
        images = find_images(student_directory)

        if len(images) < minimum_images:
            raise RuntimeError(
                f"El alumno {student_id} tiene "
                f"{len(images)} imágenes. "
                f"El mínimo requerido es {minimum_images}."
            )

        # Semilla estable y distinta por alumno.
        student_seed = (
            seed
            + sum(
                (index + 1) * ord(character)
                for index, character in enumerate(student_id)
            )
        )

        random_generator = random.Random(
            student_seed
        )

        random_generator.shuffle(images)

        (
            train_count,
            validation_count,
            test_count,
        ) = calculate_split_sizes(
            total_images=len(images),
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )

        train_end = train_count
        validation_end = (
            train_count + validation_count
        )

        # El barajado previo permite segmentar por intervalos sin solapamiento
        # y mantiene cada imagen en exactamente una de las tres particiones.
        train_images = images[:train_end]
        validation_images = images[
            train_end:validation_end
        ]
        test_images = images[validation_end:]

        copy_images(
            train_images,
            output_root / "train" / student_id,
        )

        copy_images(
            validation_images,
            output_root / "val" / student_id,
        )

        copy_images(
            test_images,
            output_root / "test" / student_id,
        )

        total_train += len(train_images)
        total_validation += len(validation_images)
        total_test += len(test_images)

        student_name = student_names.get(
            student_id,
            student_id,
        )

        manifest_students.append(
            {
                "class_index": class_index,
                "student_id": student_id,
                "name": student_name,
                "source_images": len(images),
                "train_images": len(train_images),
                "validation_images": len(
                    validation_images
                ),
                "test_images": len(test_images),
            }
        )

        print(
            f"{student_id:<12} "
            f"Total: {len(images):>3} | "
            f"Train: {len(train_images):>3} | "
            f"Val: {len(validation_images):>3} | "
            f"Test: {len(test_images):>3}"
        )

    manifest = {
        "version": 1,
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": (
                1.0
                - train_ratio
                - validation_ratio
            ),
        },
        "totals": {
            "students": len(manifest_students),
            "train_images": total_train,
            "validation_images": total_validation,
            "test_images": total_test,
            "all_images": (
                total_train
                + total_validation
                + total_test
            ),
        },
        "students": manifest_students,
    }

    manifest_path = (
        output_root / "dataset_manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 72)
    print("DATASET PREPARADO CORRECTAMENTE")
    print("=" * 72)
    print(f"Alumnos:       {len(manifest_students)}")
    print(f"Entrenamiento: {total_train}")
    print(f"Validación:    {total_validation}")
    print(f"Prueba:        {total_test}")
    print(
        f"Total:         "
        f"{total_train + total_validation + total_test}"
    )
    print(f"Manifiesto:    {manifest_path}")
    print("=" * 72)


def build_parser() -> argparse.ArgumentParser:
    """Construye el analizador de argumentos para preparar el conjunto facial."""

    parser = argparse.ArgumentParser(
        description=(
            "Divide las fotografías de alumnos en "
            "entrenamiento, validación y prueba."
        )
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=DEFAULT_RAW_ROOT,
        help="Carpeta que contiene una subcarpeta por alumno.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Carpeta donde se creará el dataset dividido.",
    )

    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Archivo students.json.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--minimum-images",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Elimina el dataset dividido anterior.",
    )

    return parser


def main() -> None:
    """Procesa las opciones de consola y ejecuta la partición del conjunto."""

    parser = build_parser()
    arguments = parser.parse_args()

    prepare_dataset(
        raw_root=arguments.raw_root.resolve(),
        output_root=arguments.output_root.resolve(),
        registry_path=arguments.registry.resolve(),
        train_ratio=arguments.train_ratio,
        validation_ratio=arguments.validation_ratio,
        minimum_images=arguments.minimum_images,
        seed=arguments.seed,
        clean_output=arguments.clean,
    )


if __name__ == "__main__":
    main()
