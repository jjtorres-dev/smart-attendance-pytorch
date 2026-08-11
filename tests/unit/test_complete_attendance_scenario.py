"""Prueba integral de una sesión con puntualidad, tardanza y ausencia."""

from datetime import date, datetime
from zoneinfo import ZoneInfo

from app.attendance.models import (
    AttendanceStatus,
    PresenceState,
)
from app.attendance.service import (
    AttendancePolicy,
    AttendanceService,
)
from app.database.attendance_repository import (
    AttendanceRepository,
)


LIMA = ZoneInfo("America/Lima")


def test_complete_classroom_scenario(
    tmp_path,
) -> None:
    """Valida el cierre conjunto de tres trayectorias de asistencia."""

    repository = AttendanceRepository(
        tmp_path / "complete_scenario.db"
    )

    repository.initialize_schema()

    repository.upsert_student(
        "A001",
        "Alumno Puntual",
    )

    repository.upsert_student(
        "A002",
        "Alumno Tardanza",
    )

    repository.upsert_student(
        "A003",
        "Alumno Ausente",
    )

    policy = AttendancePolicy(
        timezone_name="America/Lima",
        entry_deadline=datetime.strptime(
            "08:25:00",
            "%H:%M:%S",
        ).time(),
        session_end=datetime.strptime(
            "12:05:00",
            "%H:%M:%S",
        ).time(),
    )

    service = AttendanceService(
        repository,
        policy,
    )

    session_date = date(
        2026,
        8,
        4,
    )

    # A001 llega puntual.
    service.handle_entry(
        "A001",
        datetime(
            2026,
            8,
            4,
            8,
            20,
            tzinfo=LIMA,
        ),
    )

    # A001 sale temporalmente.
    service.handle_exit(
        "A001",
        datetime(
            2026,
            8,
            4,
            10,
            15,
            tzinfo=LIMA,
        ),
    )

    # A001 regresa.
    service.handle_entry(
        "A001",
        datetime(
            2026,
            8,
            4,
            10,
            25,
            tzinfo=LIMA,
        ),
    )

    # A002 llega tarde.
    service.handle_entry(
        "A002",
        datetime(
            2026,
            8,
            4,
            8,
            32,
            tzinfo=LIMA,
        ),
    )

    # A002 se retira y no vuelve.
    service.handle_exit(
        "A002",
        datetime(
            2026,
            8,
            4,
            11,
            10,
            tzinfo=LIMA,
        ),
    )

    # A003 nunca ingresa.

    service.close_session(
        session_date
    )

    a001 = repository.get_attendance(
        "A001",
        session_date,
    )

    a002 = repository.get_attendance(
        "A002",
        session_date,
    )

    a003 = repository.get_attendance(
        "A003",
        session_date,
    )

    assert a001 is not None
    assert a002 is not None
    assert a003 is not None

    # A001 fue puntual y regresó,
    # por eso termina a las 12:05.
    assert (
        a001["status"]
        == AttendanceStatus.ON_TIME.value
    )

    assert a001[
        "first_entry_at"
    ].endswith("08:20:00-05:00")

    assert a001[
        "final_exit_at"
    ].endswith("12:05:00-05:00")

    assert (
        a001["presence_state"]
        == PresenceState.CLOSED.value
    )

    # A002 llegó tarde y se fue a las 11:10.
    assert (
        a002["status"]
        == AttendanceStatus.LATE.value
    )

    assert a002[
        "first_entry_at"
    ].endswith("08:32:00-05:00")

    assert a002[
        "final_exit_at"
    ].endswith("11:10:00-05:00")

    # A003 nunca apareció.
    assert (
        a003["status"]
        == AttendanceStatus.ABSENT.value
    )

    assert a003["first_entry_at"] is None
    assert a003["final_exit_at"] is None
