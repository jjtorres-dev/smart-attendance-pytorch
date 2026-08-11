"""Construcción y ajuste de un clasificador facial basado en MobileNetV3."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    MobileNetV3,
    mobilenet_v3_small,
)


def build_face_classifier(
    number_of_classes: int,
    freeze_backbone: bool = True,
) -> tuple[MobileNetV3, MobileNet_V3_Small_Weights]:
    """
    Construye MobileNetV3 Small con pesos de ImageNet y reemplaza
    su última capa para reconocer a los alumnos registrados.
    """

    if number_of_classes < 2:
        raise ValueError(
            "Se necesitan al menos dos clases para entrenar el modelo."
        )

    weights = MobileNet_V3_Small_Weights.DEFAULT

    model = mobilenet_v3_small(
        weights=weights,
    )

    input_features = model.classifier[3].in_features

    model.classifier[3] = nn.Linear(
        input_features,
        number_of_classes,
    )

    if freeze_backbone:
        freeze_feature_extractor(model)

    return model, weights


def freeze_feature_extractor(
    model: MobileNetV3,
) -> None:
    """
    Congela las capas convolucionales y mantiene entrenable
    solamente el clasificador.
    """

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def unfreeze_last_feature_blocks(
    model: MobileNetV3,
    number_of_blocks: int = 4,
) -> None:
    """
    Mantiene congelada la mayor parte de MobileNetV3 y habilita
    para entrenamiento sus últimos bloques convolucionales.
    """

    if number_of_blocks < 1:
        raise ValueError(
            "number_of_blocks debe ser al menos 1."
        )

    for parameter in model.features.parameters():
        parameter.requires_grad = False

    blocks_to_unfreeze = model.features[-number_of_blocks:]

    for block in blocks_to_unfreeze:
        for parameter in block.parameters():
            parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def count_parameters(
    model: nn.Module,
) -> tuple[int, int]:
    """
    Devuelve parámetros totales y parámetros entrenables.
    """

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return total_parameters, trainable_parameters
