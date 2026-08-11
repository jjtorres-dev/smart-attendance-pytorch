from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class CrossingEvent(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"


@dataclass
class TrackZoneState:
    """
    Estado de zonas para un ID temporal del tracker.
    """

    confirmed_zone: str | None = None
    candidate_zone: str | None = None
    candidate_frames: int = 0

    zone_history: deque[str] = field(
        default_factory=lambda: deque(maxlen=8)
    )

    last_seen_at: float = 0.0
    last_event_at: float = -10_000.0

    def update(
        self,
        detected_zone: str | None,
        now: float,
        minimum_stable_frames: int = 5,
        event_cooldown_seconds: float = 2.0,
    ) -> CrossingEvent | None:
        """
        Actualiza el estado y devuelve ENTRY o EXIT cuando se confirma
        una secuencia completa de zonas.
        """
        self.last_seen_at = now

        if detected_zone is None:
            return None

        if detected_zone != self.candidate_zone:
            self.candidate_zone = detected_zone
            self.candidate_frames = 1
            return None

        self.candidate_frames += 1

        if self.candidate_frames < minimum_stable_frames:
            return None

        if detected_zone == self.confirmed_zone:
            return None

        self.confirmed_zone = detected_zone

        if (
            not self.zone_history
            or self.zone_history[-1] != detected_zone
        ):
            self.zone_history.append(detected_zone)

        if now - self.last_event_at < event_cooldown_seconds:
            return None

        last_zones = list(self.zone_history)[-3:]

        event: CrossingEvent | None = None

        if last_zones == ["exterior", "door", "interior"]:
            event = CrossingEvent.ENTRY

        elif last_zones == ["interior", "door", "exterior"]:
            event = CrossingEvent.EXIT

        if event is not None:
            self.last_event_at = now

            # Reiniciamos el historial dejando la zona final.
            self.zone_history.clear()
            self.zone_history.append(detected_zone)

        return event