from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceDetection:
    """Resultado normalizado de una detección facial."""

    x: int
    y: int
    width: int
    height: int
    score: float
    landmarks: tuple[tuple[int, int], ...]

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return (
            self.x + self.width // 2,
            self.y + self.height // 2,
        )


class YuNetFaceDetector:
    """Detector facial YuNet mediante la API FaceDetectorYN de OpenCV."""

    def __init__(
        self,
        model_path: str | Path,
        score_threshold: float = 0.75,
        nms_threshold: float = 0.30,
        top_k: int = 5000,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"No se encontró el modelo YuNet: {self.model_path}"
            )

        if not 0.0 < score_threshold <= 1.0:
            raise ValueError(
                "score_threshold debe encontrarse entre 0 y 1."
            )

        create_detector = getattr(
            cv2,
            "FaceDetectorYN_create",
            None,
        )

        if create_detector is not None:
            self._detector = create_detector(
                str(self.model_path),
                "",
                (320, 320),
                score_threshold,
                nms_threshold,
                top_k,
            )

        elif hasattr(cv2, "FaceDetectorYN"):
            self._detector = cv2.FaceDetectorYN.create(
                model=str(self.model_path),
                config="",
                input_size=(320, 320),
                score_threshold=score_threshold,
                nms_threshold=nms_threshold,
                top_k=top_k,
            )

        else:
            raise RuntimeError(
                "OpenCV no contiene la API FaceDetectorYN."
            )

    def detect(
        self,
        frame: np.ndarray,
    ) -> list[FaceDetection]:
        """Detecta los rostros contenidos en un fotograma BGR."""

        if frame is None or frame.size == 0:
            return []

        frame_height, frame_width = frame.shape[:2]

        self._detector.setInputSize(
            (frame_width, frame_height)
        )

        _, raw_faces = self._detector.detect(frame)

        if raw_faces is None:
            return []

        detections: list[FaceDetection] = []

        for raw_face in raw_faces:
            x, y, width, height = np.rint(
                raw_face[:4]
            ).astype(int)

            # Estructura de YuNet:
            # [x, y, w, h, 10 valores de landmarks, score]
            raw_landmarks = np.rint(
                raw_face[4:14]
            ).astype(int)

            score = float(raw_face[14])

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame_width, x + width)
            y2 = min(frame_height, y + height)

            clipped_width = x2 - x1
            clipped_height = y2 - y1

            if clipped_width <= 0 or clipped_height <= 0:
                continue

            landmarks = tuple(
                (
                    int(raw_landmarks[index]),
                    int(raw_landmarks[index + 1]),
                )
                for index in range(0, 10, 2)
            )

            detections.append(
                FaceDetection(
                    x=x1,
                    y=y1,
                    width=clipped_width,
                    height=clipped_height,
                    score=score,
                    landmarks=landmarks,
                )
            )

        detections.sort(
            key=lambda detection: detection.area,
            reverse=True,
        )

        return detections