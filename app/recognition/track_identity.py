"""Consenso temporal de identidades y asociación de rostros con personas."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from math import hypot
from typing import Mapping

import numpy as np

from app.recognition.face_detector import FaceDetection
from app.recognition.face_recognizer import RecognitionResult


@dataclass(frozen=True)
class ConfirmedIdentity:
    """Identidad consolidada a partir de múltiples observaciones de un track."""

    student_id: str
    student_name: str
    average_confidence: float
    votes: int
    total_samples: int


def calculate_track_consensus(
    history: deque[RecognitionResult],
    minimum_votes: int,
    minimum_vote_ratio: float,
) -> ConfirmedIdentity | None:
    """
    Confirma la identidad de un track mediante varias predicciones.

    Las predicciones DESCONOCIDO también forman parte del total,
    por lo que reducen el porcentaje de consenso.
    """

    if not history:
        return None

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

    student_id, votes = vote_counter.most_common(1)[0]

    vote_ratio = votes / len(history)

    # El denominador incluye también resultados desconocidos para impedir
    # que pocas predicciones conocidas dominen un historial inestable.
    if votes < minimum_votes:
        return None

    if vote_ratio < minimum_vote_ratio:
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

    latest_result = matching_results[-1]

    return ConfirmedIdentity(
        student_id=student_id,
        student_name=latest_result.student_name,
        average_confidence=average_confidence,
        votes=votes,
        total_samples=len(history),
    )


def associate_faces_to_tracks(
    faces: list[FaceDetection],
    track_boxes: Mapping[int, np.ndarray],
    expansion_ratio: float = 0.08,
) -> dict[int, FaceDetection]:
    """
    Intenta asociar cada rostro con un track de persona compatible.

    El centro del rostro debe quedar dentro de una caja de persona
    expandida. Entre las cajas compatibles se elige la de menor distancia
    normalizada al centro. Un rostro sin un track compatible puede quedar
    sin asociar.
    """

    associations: dict[int, FaceDetection] = {}

    for face in faces:
        face_center_x, face_center_y = face.center

        candidates: list[tuple[float, int]] = []

        for track_id, box in track_boxes.items():
            x1, y1, x2, y2 = map(float, box)

            width = max(x2 - x1, 1.0)
            height = max(y2 - y1, 1.0)

            expanded_x1 = x1 - width * expansion_ratio
            expanded_y1 = y1 - height * expansion_ratio
            expanded_x2 = x2 + width * expansion_ratio
            expanded_y2 = y2 + height * expansion_ratio

            # La expansión tolera pequeñas desalineaciones entre el detector
            # facial y las cajas producidas por el rastreador de personas.
            is_inside = (
                expanded_x1 <= face_center_x <= expanded_x2
                and expanded_y1 <= face_center_y <= expanded_y2
            )

            if not is_inside:
                continue

            person_center_x = (x1 + x2) / 2.0
            person_center_y = (y1 + y2) / 2.0

            diagonal = max(
                hypot(width, height),
                1.0,
            )

            normalized_distance = (
                hypot(
                    face_center_x - person_center_x,
                    face_center_y - person_center_y,
                )
                / diagonal
            )

            candidates.append(
                (normalized_distance, track_id)
            )

        if not candidates:
            continue

        _, selected_track_id = min(
            candidates,
            key=lambda candidate: candidate[0],
        )

        current_face = associations.get(
            selected_track_id
        )

        # Si más de un rostro cae en la misma persona, se conserva la
        # detección de mayor puntuación como evidencia más confiable.
        if (
            current_face is None
            or face.score > current_face.score
        ):
            associations[selected_track_id] = face

    return associations
