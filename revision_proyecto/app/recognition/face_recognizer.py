from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)

@dataclass(frozen=True)
class RecognitionResult:
    """
    Resultado de una predicción facial.
    """

    student_id: str | None
    student_name: str
    confidence: float
    margin: float
    is_unknown: bool
    predicted_class: str


class FaceRecognizer:
    """
    Carga el checkpoint entrenado de MobileNetV3 y reconoce
    rostros previamente recortados por YuNet.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        confidence_threshold: float = 0.85,
        margin_threshold: float = 0.30,
        device: str | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"No se encontró el checkpoint: "
                f"{self.checkpoint_path}"
            )

        if not 0.0 < confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold debe estar entre 0 y 1."
            )

        if not 0.0 <= margin_threshold <= 1.0:
            raise ValueError(
                "margin_threshold debe estar entre 0 y 1."
            )

        if device is None:
            self.device = torch.device(
                "cuda:0"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        self.classes = [
            str(class_id)
            for class_id in checkpoint["classes"]
        ]

        self.student_names = {
            str(student_id): str(name)
            for student_id, name
            in checkpoint.get(
                "student_names",
                {},
            ).items()
        }

        number_of_classes = int(
            checkpoint["number_of_classes"]
        )

        if number_of_classes != len(self.classes):
            raise RuntimeError(
                "El número de clases del checkpoint "
                "no coincide con la lista de clases."
            )

        self.model = mobilenet_v3_small(
            weights=None,
        )

        input_features = (
            self.model.classifier[3].in_features
        )

        self.model.classifier[3] = nn.Linear(
            input_features,
            number_of_classes,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        input_size = int(
            checkpoint.get("input_size", 224)
        )

        normalization = checkpoint.get(
            "normalization",
            {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        )

        weights = MobileNet_V3_Small_Weights.DEFAULT

        self.transform = transforms.Compose(
            [
                # Las imágenes guardadas durante el registro facial
                # fueron llevadas primero a 224x224.
                transforms.Resize(
                    (input_size, input_size)
                ),

                # Después aplicamos exactamente la misma transformación
                # utilizada durante validación y test.
                weights.transforms(),
            ]
        )



    @property
    def device_name(self) -> str:
        if self.device.type == "cuda":
            return torch.cuda.get_device_name(
                self.device.index or 0
            )

        return "CPU"

    def recognize(
        self,
        face_bgr: np.ndarray,
    ) -> RecognitionResult:
        """
        Reconoce un recorte facial en formato BGR.
        """

        if face_bgr is None or face_bgr.size == 0:
            raise ValueError(
                "El recorte facial está vacío."
            )

        face_rgb = cv2.cvtColor(
            face_bgr,
            cv2.COLOR_BGR2RGB,
        )

        pil_image = Image.fromarray(face_rgb)

        input_tensor = self.transform(
            pil_image
        ).unsqueeze(0)

        input_tensor = input_tensor.to(
            self.device,
            non_blocking=True,
        )

        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    logits = self.model(
                        input_tensor
                    )
            else:
                logits = self.model(
                    input_tensor
                )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0]

        number_of_results = min(
            2,
            len(self.classes),
        )

        top_values, top_indices = torch.topk(
            probabilities,
            k=number_of_results,
        )

        confidence = float(
            top_values[0].item()
        )

        predicted_index = int(
            top_indices[0].item()
        )

        second_confidence = (
            float(top_values[1].item())
            if number_of_results > 1
            else 0.0
        )

        margin = (
            confidence - second_confidence
        )

        predicted_class = self.classes[
            predicted_index
        ]

        is_unknown = (
            confidence < self.confidence_threshold
            or margin < self.margin_threshold
        )

        if is_unknown:
            return RecognitionResult(
                student_id=None,
                student_name="DESCONOCIDO",
                confidence=confidence,
                margin=margin,
                is_unknown=True,
                predicted_class=predicted_class,
            )

        student_name = self.student_names.get(
            predicted_class,
            predicted_class,
        )

        return RecognitionResult(
            student_id=predicted_class,
            student_name=student_name,
            confidence=confidence,
            margin=margin,
            is_unknown=False,
            predicted_class=predicted_class,
        )