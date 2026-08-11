from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class AttendanceStatus(str, Enum):
    PENDING = "PENDING"
    ON_TIME = "ON_TIME"
    LATE = "LATE"
    ABSENT = "ABSENT"


class PresenceState(str, Enum):
    NOT_ENTERED = "NOT_ENTERED"
    INSIDE = "INSIDE"
    OUTSIDE_TEMPORARY = "OUTSIDE_TEMPORARY"
    CLOSED = "CLOSED"


class AttendanceEventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    REENTRY = "REENTRY"
    SESSION_CLOSE = "SESSION_CLOSE"


@dataclass(frozen=True)
class AttendanceActionResult:
    accepted: bool
    student_id: str
    action: str
    occurred_at: datetime
    message: str