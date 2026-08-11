"""Pruebas del consenso de identidad y asociación rostro-persona."""

from collections import deque

import numpy as np

from app.recognition.face_detector import FaceDetection
from app.recognition.face_recognizer import RecognitionResult
from app.recognition.track_identity import (
    associate_faces_to_tracks,
    calculate_track_consensus,
)


def known_result(
    student_id: str,
    name: str,
    confidence: float = 0.95,
) -> RecognitionResult:
    """Construye una predicción conocida para escenarios de consenso."""

    return RecognitionResult(
        student_id=student_id,
        student_name=name,
        confidence=confidence,
        margin=0.80,
        is_unknown=False,
        predicted_class=student_id,
    )


def unknown_result() -> RecognitionResult:
    """Construye una predicción rechazada por confianza o margen."""

    return RecognitionResult(
        student_id=None,
        student_name="DESCONOCIDO",
        confidence=0.60,
        margin=0.10,
        is_unknown=True,
        predicted_class="A001",
    )


def test_confirms_identity_with_enough_votes() -> None:
    """Confirma al estudiante que alcanza votos y proporción mínimos."""

    history = deque(
        [
            known_result("A001", "Alumno Uno"),
            known_result("A001", "Alumno Uno"),
            known_result("A001", "Alumno Uno"),
            known_result("A001", "Alumno Uno"),
            known_result("A001", "Alumno Uno"),
            known_result("A001", "Alumno Uno"),
            unknown_result(),
            unknown_result(),
        ],
        maxlen=12,
    )

    consensus = calculate_track_consensus(
        history=history,
        minimum_votes=6,
        minimum_vote_ratio=0.60,
    )

    assert consensus is not None
    assert consensus.student_id == "A001"
    assert consensus.votes == 6


def test_rejects_unstable_identity() -> None:
    """Rechaza un historial dividido que no alcanza consenso suficiente."""

    history = deque(
        [
            known_result("A001", "Alumno Uno"),
            known_result("A001", "Alumno Uno"),
            known_result("A001", "Alumno Uno"),
            known_result("A002", "Alumno Dos"),
            known_result("A002", "Alumno Dos"),
            unknown_result(),
            unknown_result(),
            unknown_result(),
        ],
        maxlen=12,
    )

    consensus = calculate_track_consensus(
        history=history,
        minimum_votes=4,
        minimum_vote_ratio=0.60,
    )

    assert consensus is None


def test_associates_face_with_correct_track() -> None:
    """Asocia cada rostro con la caja espacial de la persona correcta."""

    track_boxes = {
        10: np.array([100, 100, 400, 700]),
        20: np.array([500, 100, 800, 700]),
    }

    faces = [
        FaceDetection(
            x=170,
            y=150,
            width=100,
            height=120,
            score=0.95,
            landmarks=(),
        ),
        FaceDetection(
            x=570,
            y=150,
            width=100,
            height=120,
            score=0.96,
            landmarks=(),
        ),
    ]

    associations = associate_faces_to_tracks(
        faces=faces,
        track_boxes=track_boxes,
    )

    assert associations[10] == faces[0]
    assert associations[20] == faces[1]
