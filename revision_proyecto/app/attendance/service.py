from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from app.attendance.models import (
    AttendanceActionResult,
    AttendanceEventType,
    AttendanceStatus,
    PresenceState,
)
from app.database.attendance_repository import (
    AttendanceRepository,
)


@dataclass(frozen=True)
class AttendancePolicy:
    timezone_name: str
    entry_deadline: time
    session_end: time

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> "AttendancePolicy":
        config_path = Path(path)

        with config_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        return cls(
            timezone_name=str(data["timezone"]),
            entry_deadline=time.fromisoformat(
                data["entry_deadline"]
            ),
            session_end=time.fromisoformat(
                data["session_end"]
            ),
        )


class AttendanceService:
    """
    Aplica las reglas de ingreso, salida, reingreso
    y cierre de la sesión.
    """

    def __init__(
        self,
        repository: AttendanceRepository,
        policy: AttendancePolicy,
    ) -> None:
        self.repository = repository
        self.policy = policy
        self.timezone = ZoneInfo(
            policy.timezone_name
        )

    def _localize(
        self,
        occurred_at: datetime,
    ) -> datetime:
        if occurred_at.tzinfo is None:
            return occurred_at.replace(
                tzinfo=self.timezone
            )

        return occurred_at.astimezone(
            self.timezone
        )

    def _deadline_for(
        self,
        session_date: date,
    ) -> datetime:
        return datetime.combine(
            session_date,
            self.policy.entry_deadline,
            tzinfo=self.timezone,
        )

    def _end_for(
        self,
        session_date: date,
    ) -> datetime:
        return datetime.combine(
            session_date,
            self.policy.session_end,
            tzinfo=self.timezone,
        )

    def handle_entry(
        self,
        student_id: str,
        occurred_at: datetime,
        track_id: int | None = None,
    ) -> AttendanceActionResult:
        local_time = self._localize(
            occurred_at
        )

        session_date = local_time.date()
        session_end = self._end_for(
            session_date
        )

        if local_time >= session_end:
            return AttendanceActionResult(
                accepted=False,
                student_id=student_id,
                action="ENTRY_IGNORED",
                occurred_at=local_time,
                message=(
                    "La sesión ya terminó."
                ),
            )

        self.repository.ensure_attendance(
            student_id,
            session_date,
        )

        attendance = (
            self.repository.get_attendance(
                student_id,
                session_date,
            )
        )

        if attendance is None:
            raise RuntimeError(
                "No se pudo crear la asistencia."
            )

        current_state = PresenceState(
            attendance["presence_state"]
        )

        if current_state == PresenceState.CLOSED:
            return AttendanceActionResult(
                accepted=False,
                student_id=student_id,
                action="ENTRY_IGNORED",
                occurred_at=local_time,
                message="La asistencia está cerrada.",
            )

        first_entry = attendance[
            "first_entry_at"
        ]

        if first_entry is None:
            deadline = self._deadline_for(
                session_date
            )

            status = (
                AttendanceStatus.ON_TIME
                if local_time <= deadline
                else AttendanceStatus.LATE
            )

            self.repository.update_attendance(
                student_id,
                session_date,
                status=status.value,
                presence_state=(
                    PresenceState.INSIDE.value
                ),
                first_entry_at=local_time.isoformat(
                    timespec="seconds"
                ),
                candidate_exit_at=None,
                final_exit_at=None,
            )

            self.repository.add_event(
                student_id=student_id,
                session_date=session_date,
                event_type=(
                    AttendanceEventType.ENTRY.value
                ),
                occurred_at=local_time,
                track_id=track_id,
                details=status.value,
            )

            return AttendanceActionResult(
                accepted=True,
                student_id=student_id,
                action="ENTRY",
                occurred_at=local_time,
                message=(
                    "Entrada puntual."
                    if status == AttendanceStatus.ON_TIME
                    else "Entrada con tardanza."
                ),
            )

        if (
            current_state
            == PresenceState.OUTSIDE_TEMPORARY
        ):
            self.repository.update_attendance(
                student_id,
                session_date,
                presence_state=(
                    PresenceState.INSIDE.value
                ),
                candidate_exit_at=None,
            )

            self.repository.add_event(
                student_id=student_id,
                session_date=session_date,
                event_type=(
                    AttendanceEventType.REENTRY.value
                ),
                occurred_at=local_time,
                track_id=track_id,
            )

            return AttendanceActionResult(
                accepted=True,
                student_id=student_id,
                action="REENTRY",
                occurred_at=local_time,
                message="Reingreso registrado.",
            )

        return AttendanceActionResult(
            accepted=False,
            student_id=student_id,
            action="ENTRY_DUPLICATE",
            occurred_at=local_time,
            message=(
                "El alumno ya se encuentra dentro."
            ),
        )

    def handle_exit(
        self,
        student_id: str,
        occurred_at: datetime,
        track_id: int | None = None,
    ) -> AttendanceActionResult:
        local_time = self._localize(
            occurred_at
        )

        session_date = local_time.date()

        self.repository.ensure_attendance(
            student_id,
            session_date,
        )

        attendance = (
            self.repository.get_attendance(
                student_id,
                session_date,
            )
        )

        if attendance is None:
            raise RuntimeError(
                "No se encontró la asistencia."
            )

        current_state = PresenceState(
            attendance["presence_state"]
        )

        if attendance["first_entry_at"] is None:
            return AttendanceActionResult(
                accepted=False,
                student_id=student_id,
                action="EXIT_WITHOUT_ENTRY",
                occurred_at=local_time,
                message=(
                    "No existe una entrada previa."
                ),
            )

        if current_state == PresenceState.INSIDE:
            self.repository.update_attendance(
                student_id,
                session_date,
                presence_state=(
                    PresenceState
                    .OUTSIDE_TEMPORARY
                    .value
                ),
                candidate_exit_at=(
                    local_time.isoformat(
                        timespec="seconds"
                    )
                ),
            )

            self.repository.add_event(
                student_id=student_id,
                session_date=session_date,
                event_type=(
                    AttendanceEventType.EXIT.value
                ),
                occurred_at=local_time,
                track_id=track_id,
            )

            return AttendanceActionResult(
                accepted=True,
                student_id=student_id,
                action="EXIT",
                occurred_at=local_time,
                message=(
                    "Salida provisional registrada."
                ),
            )

        return AttendanceActionResult(
            accepted=False,
            student_id=student_id,
            action="EXIT_DUPLICATE",
            occurred_at=local_time,
            message=(
                "El alumno ya estaba fuera."
            ),
        )

    def close_session(
        self,
        session_date: date,
    ) -> dict[str, int]:
        close_time = self._end_for(
            session_date
        )

        summary = {
            "TOTAL": 0,
            "ON_TIME": 0,
            "LATE": 0,
            "ABSENT": 0,
        }

        for student in (
            self.repository.list_active_students()
        ):
            student_id = student["student_id"]

            self.repository.ensure_attendance(
                student_id,
                session_date,
            )

            attendance = (
                self.repository.get_attendance(
                    student_id,
                    session_date,
                )
            )

            if attendance is None:
                continue

            current_state = PresenceState(
                attendance["presence_state"]
            )

            if current_state == PresenceState.CLOSED:
                status = attendance["status"]

                summary["TOTAL"] += 1

                if status in summary:
                    summary[status] += 1

                continue

            if attendance["first_entry_at"] is None:
                status = AttendanceStatus.ABSENT
                final_exit = None

            else:
                status = AttendanceStatus(
                    attendance["status"]
                )

                if (
                    current_state
                    == PresenceState.OUTSIDE_TEMPORARY
                ):
                    final_exit = attendance[
                        "candidate_exit_at"
                    ]
                else:
                    final_exit = close_time.isoformat(
                        timespec="seconds"
                    )

            self.repository.update_attendance(
                student_id,
                session_date,
                status=status.value,
                presence_state=(
                    PresenceState.CLOSED.value
                ),
                final_exit_at=final_exit,
            )

            self.repository.add_event(
                student_id=student_id,
                session_date=session_date,
                event_type=(
                    AttendanceEventType
                    .SESSION_CLOSE
                    .value
                ),
                occurred_at=close_time,
                details=status.value,
            )

            summary["TOTAL"] += 1
            summary[status.value] += 1

        return summary