"""Entrenamiento, selección y evaluación de un clasificador facial."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Small_Weights

from app.recognition.face_classifier import (
    build_face_classifier,
    count_parameters,
    unfreeze_last_feature_blocks,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "students"
    / "dataset"
)

DEFAULT_REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "students"
    / "students.json"
)

DEFAULT_MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "recognition"
)


def set_random_seed(seed: int) -> None:
    """
    Establece semillas para mejorar la repetibilidad de los experimentos.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


def load_student_names(
    registry_path: Path,
) -> dict[str, str]:
    """
    Carga el nombre completo asociado con cada código.
    """

    if not registry_path.exists():
        return {}

    with registry_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return {
        str(student["student_id"]): str(student["name"])
        for student in data.get("students", [])
        if "student_id" in student and "name" in student
    }


def build_transforms() -> tuple[Any, Any]:
    """
    Crea transformaciones de entrenamiento y evaluación.
    """

    weights = MobileNet_V3_Small_Weights.DEFAULT

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.82, 1.0),
                ratio=(0.90, 1.10),
            ),
            transforms.RandomHorizontalFlip(
                p=0.5,
            ),
            transforms.RandomRotation(
                degrees=8,
            ),
            transforms.ColorJitter(
                brightness=0.20,
                contrast=0.20,
                saturation=0.12,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    evaluation_transform = weights.transforms()

    return train_transform, evaluation_transform


def build_datasets(
    dataset_root: Path,
) -> tuple[
    datasets.ImageFolder,
    datasets.ImageFolder,
    datasets.ImageFolder,
]:
    """Carga las particiones y comprueba que compartan las mismas clases."""

    train_directory = dataset_root / "train"
    validation_directory = dataset_root / "val"
    test_directory = dataset_root / "test"

    for directory in (
        train_directory,
        validation_directory,
        test_directory,
    ):
        if not directory.exists():
            raise FileNotFoundError(
                f"No se encontró el directorio: {directory}"
            )

    train_transform, evaluation_transform = build_transforms()

    train_dataset = datasets.ImageFolder(
        root=train_directory,
        transform=train_transform,
    )

    validation_dataset = datasets.ImageFolder(
        root=validation_directory,
        transform=evaluation_transform,
    )

    test_dataset = datasets.ImageFolder(
        root=test_directory,
        transform=evaluation_transform,
    )

    expected_classes = train_dataset.classes

    if validation_dataset.classes != expected_classes:
        raise RuntimeError(
            "Las clases de validación no coinciden con entrenamiento."
        )

    if test_dataset.classes != expected_classes:
        raise RuntimeError(
            "Las clases de prueba no coinciden con entrenamiento."
        )

    if len(expected_classes) < 2:
        raise RuntimeError(
            "Se necesitan al menos dos alumnos."
        )

    return (
        train_dataset,
        validation_dataset,
        test_dataset,
    )


def build_data_loaders(
    train_dataset: datasets.ImageFolder,
    validation_dataset: datasets.ImageFolder,
    test_dataset: datasets.ImageFolder,
    batch_size: int,
    number_of_workers: int,
    use_cuda: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construye cargadores con barajado exclusivo para entrenamiento."""

    common_arguments = {
        "batch_size": batch_size,
        "num_workers": number_of_workers,
        "pin_memory": use_cuda,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=False,
        **common_arguments,
    )

    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **common_arguments,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        **common_arguments,
    )

    return train_loader, validation_loader, test_loader


def create_optimizer(
    model: nn.Module,
    head_learning_rate: float,
    backbone_learning_rate: float,
    weight_decay: float,
    backbone_enabled: bool,
) -> AdamW:
    """
    Crea grupos de parámetros con tasas de aprendizaje diferentes.
    """

    parameter_groups: list[dict[str, Any]] = [
        {
            "params": [
                parameter
                for parameter in model.classifier.parameters()
                if parameter.requires_grad
            ],
            "lr": head_learning_rate,
        }
    ]

    if backbone_enabled:
        backbone_parameters = [
            parameter
            for parameter in model.features.parameters()
            if parameter.requires_grad
        ]

        if backbone_parameters:
            parameter_groups.append(
                {
                    "params": backbone_parameters,
                    "lr": backbone_learning_rate,
                }
            )

    return AdamW(
        parameter_groups,
        weight_decay=weight_decay,
    )


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
) -> tuple[float, float, list[int], list[int]]:
    """Ejecuta una época de entrenamiento o evaluación y reúne sus métricas."""

    training = optimizer is not None

    if training:
        model.train()
        gradient_context = torch.enable_grad()
    else:
        # En evaluación se desactivan gradientes y capas estocásticas para
        # medir el modelo sin actualizar ni retener su grafo computacional.
        model.eval()
        gradient_context = torch.inference_mode()

    running_loss = 0.0
    correct_predictions = 0
    total_examples = 0

    all_targets: list[int] = []
    all_predictions: list[int] = []

    with gradient_context:
        for inputs, targets in data_loader:
            inputs = inputs.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                non_blocking=True,
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True,
                )

            autocast_context = (
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                )
                if use_amp
                else nullcontext()
            )

            with autocast_context:
                logits = model(inputs)
                loss = criterion(logits, targets)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()

                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=5.0,
                    )

                    scaler.step(optimizer)
                    scaler.update()

                else:
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=5.0,
                    )

                    optimizer.step()

            predictions = logits.argmax(dim=1)

            batch_size = targets.size(0)

            running_loss += (
                loss.detach().item() * batch_size
            )

            # La pérdida se pondera por tamaño de lote para que el último lote,
            # potencialmente menor, no tenga el mismo peso que uno completo.
            correct_predictions += (
                predictions == targets
            ).sum().item()

            total_examples += batch_size

            all_targets.extend(
                targets.detach().cpu().tolist()
            )

            all_predictions.extend(
                predictions.detach().cpu().tolist()
            )

    average_loss = running_loss / max(total_examples, 1)

    accuracy = (
        correct_predictions / max(total_examples, 1)
    )

    return (
        average_loss,
        accuracy,
        all_targets,
        all_predictions,
    )


def save_training_history(
    history: list[dict[str, float | int]],
    output_directory: Path,
) -> None:
    """Guarda el historial y grafica pérdidas y exactitudes por época."""

    history_path = output_directory / "training_history.json"

    with history_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            ensure_ascii=False,
            indent=2,
        )

    epochs = [
        int(item["epoch"])
        for item in history
    ]

    training_losses = [
        float(item["train_loss"])
        for item in history
    ]

    validation_losses = [
        float(item["validation_loss"])
        for item in history
    ]

    plt.figure(figsize=(9, 5))
    plt.plot(
        epochs,
        training_losses,
        label="Entrenamiento",
    )
    plt.plot(
        epochs,
        validation_losses,
        label="Validación",
    )
    plt.xlabel("Época")
    plt.ylabel("Pérdida")
    plt.title("Evolución de la pérdida")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_directory / "loss_history.png",
        dpi=160,
    )
    plt.close()

    training_accuracies = [
        float(item["train_accuracy"]) * 100
        for item in history
    ]

    validation_accuracies = [
        float(item["validation_accuracy"]) * 100
        for item in history
    ]

    plt.figure(figsize=(9, 5))
    plt.plot(
        epochs,
        training_accuracies,
        label="Entrenamiento",
    )
    plt.plot(
        epochs,
        validation_accuracies,
        label="Validación",
    )
    plt.xlabel("Época")
    plt.ylabel("Exactitud (%)")
    plt.title("Evolución de la exactitud")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_directory / "accuracy_history.png",
        dpi=160,
    )
    plt.close()


def save_test_results(
    targets: list[int],
    predictions: list[int],
    class_ids: list[str],
    student_names: dict[str, str],
    test_loss: float,
    test_accuracy: float,
    output_directory: Path,
) -> None:
    """Guarda matriz de confusión, métricas por clase y resultado de prueba."""

    display_names = [
        (
            f"{student_id}\n"
            f"{student_names.get(student_id, student_id)}"
        )
        for student_id in class_ids
    ]

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(range(len(class_ids))),
    )

    figure, axis = plt.subplots(
        figsize=(8, 7)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=display_names,
    )

    display.plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )

    axis.set_title(
        "Matriz de confusión — conjunto de prueba"
    )

    figure.tight_layout()

    figure.savefig(
        output_directory / "confusion_matrix.png",
        dpi=170,
    )

    plt.close(figure)

    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(class_ids))),
        target_names=class_ids,
        output_dict=True,
        zero_division=0,
    )

    results = {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }

    with (
        output_directory / "test_results.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )


def train(args: argparse.Namespace) -> None:
    """Orquesta ajuste por etapas, selección del modelo y evaluación final."""

    set_random_seed(args.seed)

    dataset_root = args.dataset_root.resolve()
    registry_path = args.registry.resolve()
    output_directory = args.output_directory.resolve()

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    use_amp = (
        args.mixed_precision
        and device.type == "cuda"
    )

    (
        train_dataset,
        validation_dataset,
        test_dataset,
    ) = build_datasets(dataset_root)

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = build_data_loaders(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        number_of_workers=args.workers,
        use_cuda=device.type == "cuda",
    )

    number_of_classes = len(
        train_dataset.classes
    )

    model, _ = build_face_classifier(
        number_of_classes=number_of_classes,
        freeze_backbone=args.freeze_epochs > 0,
    )

    model.to(device)

    total_parameters, trainable_parameters = (
        count_parameters(model)
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=args.label_smoothing,
    )

    backbone_enabled = args.freeze_epochs == 0

    if backbone_enabled:
        unfreeze_last_feature_blocks(
            model,
            number_of_blocks=args.unfreeze_blocks,
        )

    optimizer = create_optimizer(
        model=model,
        head_learning_rate=args.head_learning_rate,
        backbone_learning_rate=(
            args.backbone_learning_rate
        ),
        weight_decay=args.weight_decay,
        backbone_enabled=backbone_enabled,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.30,
        patience=2,
        min_lr=1e-6,
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )
        if use_amp
        else None
    )

    student_names = load_student_names(
        registry_path
    )

    checkpoint_path = (
        output_directory
        / "mobilenet_v3_face_best.pth"
    )

    history: list[dict[str, float | int]] = []

    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    print("=" * 76)
    print("ENTRENAMIENTO DEL RECONOCEDOR FACIAL")
    print("=" * 76)
    print(f"Dispositivo:          {device_name}")
    print(f"PyTorch:              {torch.__version__}")
    print(f"Torchvision:          {torchvision.__version__}")
    print(f"Precisión mixta:      {use_amp}")
    print(f"Alumnos:              {train_dataset.classes}")
    print(f"Entrenamiento:        {len(train_dataset)}")
    print(f"Validación:           {len(validation_dataset)}")
    print(f"Prueba:               {len(test_dataset)}")
    print(f"Batch size:           {args.batch_size}")
    print(f"Épocas máximas:       {args.epochs}")
    print(f"Épocas congeladas:    {args.freeze_epochs}")
    print(f"Parámetros totales:   {total_parameters:,}")
    print(f"Entrenables iniciales:{trainable_parameters:,}")
    print("=" * 76)

    training_started_at = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        if (
            args.freeze_epochs > 0
            and epoch == args.freeze_epochs + 1
        ):
            # La primera etapa aprende el clasificador; durante el ajuste
            # fino, el backbone y el clasificador pueden utilizar tasas de
            # aprendizaje diferenciadas.
            print(
                "\nDescongelando los últimos "
                f"{args.unfreeze_blocks} bloques..."
            )

            unfreeze_last_feature_blocks(
                model,
                number_of_blocks=args.unfreeze_blocks,
            )

            optimizer = create_optimizer(
                model=model,
                head_learning_rate=(
                    args.head_learning_rate * 0.5
                ),
                backbone_learning_rate=(
                    args.backbone_learning_rate
                ),
                weight_decay=args.weight_decay,
                backbone_enabled=True,
            )

            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.30,
                patience=2,
                min_lr=1e-6,
            )

            (
                _,
                trainable_parameters,
            ) = count_parameters(model)

            print(
                "Parámetros entrenables después "
                f"del desbloqueo: {trainable_parameters:,}\n"
            )

        epoch_started_at = time.perf_counter()

        (
            train_loss,
            train_accuracy,
            _,
            _,
        ) = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
        )

        (
            validation_loss,
            validation_accuracy,
            _,
            _,
        ) = run_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
        )

        scheduler.step(validation_loss)

        current_learning_rates = [
            group["lr"]
            for group in optimizer.param_groups
        ]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": (
                    validation_accuracy
                ),
                "learning_rate": max(
                    current_learning_rates
                ),
            }
        )

        epoch_seconds = (
            time.perf_counter()
            - epoch_started_at
        )

        print(
            f"Época {epoch:02d}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train acc: {train_accuracy * 100:6.2f}% | "
            f"Val loss: {validation_loss:.4f} | "
            f"Val acc: {validation_accuracy * 100:6.2f}% | "
            f"LR: {max(current_learning_rates):.2e} | "
            f"{epoch_seconds:.1f}s"
        )

        improved = (
            validation_loss
            < best_validation_loss - args.minimum_delta
        )

        if improved:
            # La selección usa exclusivamente validación; el conjunto de
            # prueba permanece reservado para la medición final sin sesgo.
            best_validation_loss = validation_loss
            best_validation_accuracy = (
                validation_accuracy
            )
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "architecture": "mobilenet_v3_small",
                "model_state_dict": copy.deepcopy(
                    model.state_dict()
                ),
                "class_to_idx": (
                    train_dataset.class_to_idx
                ),
                "classes": train_dataset.classes,
                "student_names": student_names,
                "number_of_classes": (
                    number_of_classes
                ),
                "input_size": 224,
                "normalization": {
                    "mean": [
                        0.485,
                        0.456,
                        0.406,
                    ],
                    "std": [
                        0.229,
                        0.224,
                        0.225,
                    ],
                },
                "best_epoch": best_epoch,
                "best_validation_loss": (
                    best_validation_loss
                ),
                "best_validation_accuracy": (
                    best_validation_accuracy
                ),
                "created_at": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "torch_version": torch.__version__,
                "torchvision_version": (
                    torchvision.__version__
                ),
            }

            torch.save(
                checkpoint,
                checkpoint_path,
            )

            print(
                f"  → Mejor modelo guardado: "
                f"{checkpoint_path.name}"
            )

        else:
            epochs_without_improvement += 1

            print(
                "  → Sin mejora: "
                f"{epochs_without_improvement}/"
                f"{args.early_stopping_patience}"
            )

        if (
            epochs_without_improvement
            >= args.early_stopping_patience
        ):
            print(
                "\nEarly stopping activado."
            )
            break

    total_training_seconds = (
        time.perf_counter()
        - training_started_at
    )

    if not checkpoint_path.exists():
        raise RuntimeError(
            "No se generó el checkpoint del modelo."
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # La prueba se realiza con el mejor checkpoint de validación y no
    # necesariamente con los parámetros de la última época ejecutada.
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    (
        test_loss,
        test_accuracy,
        test_targets,
        test_predictions,
    ) = run_epoch(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scaler=None,
        use_amp=use_amp,
    )

    save_training_history(
        history=history,
        output_directory=output_directory,
    )

    save_test_results(
        targets=test_targets,
        predictions=test_predictions,
        class_ids=train_dataset.classes,
        student_names=student_names,
        test_loss=test_loss,
        test_accuracy=test_accuracy,
        output_directory=output_directory,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_validation_loss": (
            best_validation_loss
        ),
        "best_validation_accuracy": (
            best_validation_accuracy
        ),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "training_seconds": (
            total_training_seconds
        ),
        "checkpoint": str(checkpoint_path),
    }

    with (
        output_directory / "training_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 76)
    print("ENTRENAMIENTO FINALIZADO")
    print("=" * 76)
    print(f"Mejor época:          {best_epoch}")
    print(
        f"Mejor validación:     "
        f"{best_validation_accuracy * 100:.2f}%"
    )
    print(f"Pérdida en test:      {test_loss:.4f}")
    print(
        f"Exactitud en test:    "
        f"{test_accuracy * 100:.2f}%"
    )
    print(
        f"Tiempo total:         "
        f"{total_training_seconds:.1f}s"
    )
    print(f"Modelo:               {checkpoint_path}")
    print(
        "Matriz de confusión: "
        f"{output_directory / 'confusion_matrix.png'}"
    )
    print("=" * 76)


def build_parser() -> argparse.ArgumentParser:
    """Construye el analizador de hiperparámetros y rutas de entrenamiento."""

    parser = argparse.ArgumentParser(
        description=(
            "Entrena MobileNetV3 Small para reconocer alumnos."
        )
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )

    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_MODEL_DIRECTORY,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--unfreeze-blocks",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "En Windows se recomienda empezar con 0."
        ),
    )

    parser.add_argument(
        "--head-learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--minimum-delta",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--mixed-precision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    return parser


def main() -> None:
    """Valida los hiperparámetros básicos e inicia el entrenamiento."""

    parser = build_parser()
    arguments = parser.parse_args()

    if arguments.epochs < 1:
        parser.error("--epochs debe ser al menos 1.")

    if arguments.freeze_epochs < 0:
        parser.error(
            "--freeze-epochs no puede ser negativo."
        )

    if arguments.batch_size < 1:
        parser.error(
            "--batch-size debe ser al menos 1."
        )

    train(arguments)


if __name__ == "__main__":
    main()
