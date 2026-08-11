"""Tipos de dominio utilizados para representar la asistencia estudiantil."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class AttendanceStatus(str, Enum):
    """Clasificación académica de la asistencia de un estudiante."""

    PENDING = "PENDING"
    ON_TIME = "ON_TIME"
    LATE = "LATE"
    ABSENT = "ABSENT"


class PresenceState(str, Enum):
    """Estado de presencia del estudiante durante una sesión."""

    NOT_ENTERED = "NOT_ENTERED"
    INSIDE = "INSIDE"
    OUTSIDE_TEMPORARY = "OUTSIDE_TEMPORARY"
    CLOSED = "CLOSED"


class AttendanceEventType(str, Enum):
    """Tipos de transiciones registradas en el historial de asistencia."""

    ENTRY = "ENTRY"
    EXIT = "EXIT"
    REENTRY = "REENTRY"
    SESSION_CLOSE = "SESSION_CLOSE"


@dataclass(frozen=True)
class AttendanceActionResult:
    """Resultado inmutable de procesar una entrada, salida o reingreso."""

    accepted: bool
    student_id: str
    action: str
    occurred_at: datetime
    message: str
