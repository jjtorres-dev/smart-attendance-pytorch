"""Pruebas unitarias de las transiciones del servicio de asistencia."""

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


def create_service(
    tmp_path,
) -> tuple[
    AttendanceService,
    AttendanceRepository,
]:
    """Crea un servicio aislado con dos estudiantes y política fija."""

    repository = AttendanceRepository(
        tmp_path / "attendance_test.db"
    )

    repository.initialize_schema()

    repository.upsert_student(
        "A001",
        "Alumno Uno",
    )

    repository.upsert_student(
        "A002",
        "Alumno Dos",
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

    return service, repository


def test_registers_on_time_entry(
    tmp_path,
) -> None:
    """Comprueba que una entrada anterior al límite se marque puntual."""

    service, repository = create_service(
        tmp_path
    )

    occurred_at = datetime(
        2026,
        7,
        30,
        8,
        20,
        tzinfo=LIMA,
    )

    result = service.handle_entry(
        "A001",
        occurred_at,
    )

    attendance = repository.get_attendance(
        "A001",
        occurred_at.date(),
    )

    assert result.accepted is True
    assert result.action == "ENTRY"
    assert attendance is not None
    assert (
        attendance["status"]
        == AttendanceStatus.ON_TIME.value
    )
    assert (
        attendance["presence_state"]
        == PresenceState.INSIDE.value
    )


def test_registers_late_entry(
    tmp_path,
) -> None:
    """Comprueba que una entrada posterior al límite se marque tardía."""

    service, repository = create_service(
        tmp_path
    )

    occurred_at = datetime(
        2026,
        7,
        30,
        8,
        30,
        tzinfo=LIMA,
    )

    service.handle_entry(
        "A001",
        occurred_at,
    )

    attendance = repository.get_attendance(
        "A001",
        occurred_at.date(),
    )

    assert attendance is not None
    assert (
        attendance["status"]
        == AttendanceStatus.LATE.value
    )


def test_exit_and_reentry_finishes_at_session_end(
    tmp_path,
) -> None:
    """Verifica que un reingreso descarte la salida provisional."""

    service, repository = create_service(
        tmp_path
    )

    session_date = date(
        2026,
        7,
        30,
    )

    service.handle_entry(
        "A001",
        datetime(
            2026,
            7,
            30,
            8,
            20,
            tzinfo=LIMA,
        ),
    )

    service.handle_exit(
        "A001",
        datetime(
            2026,
            7,
            30,
            10,
            0,
            tzinfo=LIMA,
        ),
    )

    service.handle_entry(
        "A001",
        datetime(
            2026,
            7,
            30,
            10,
            10,
            tzinfo=LIMA,
        ),
    )

    service.close_session(
        session_date
    )

    attendance = repository.get_attendance(
        "A001",
        session_date,
    )

    assert attendance is not None
    assert attendance[
        "candidate_exit_at"
    ] is None
    assert attendance[
        "final_exit_at"
    ].endswith("12:05:00-05:00")


def test_exit_without_return_becomes_final_exit(
    tmp_path,
) -> None:
    """Verifica que una salida sin retorno se consolide al cerrar."""

    service, repository = create_service(
        tmp_path
    )

    session_date = date(
        2026,
        7,
        30,
    )

    service.handle_entry(
        "A001",
        datetime(
            2026,
            7,
            30,
            8,
            20,
            tzinfo=LIMA,
        ),
    )

    service.handle_exit(
        "A001",
        datetime(
            2026,
            7,
            30,
            10,
            15,
            tzinfo=LIMA,
        ),
    )

    service.close_session(
        session_date
    )

    attendance = repository.get_attendance(
        "A001",
        session_date,
    )

    assert attendance is not None
    assert attendance[
        "final_exit_at"
    ].endswith("10:15:00-05:00")


def test_student_without_entry_is_absent(
    tmp_path,
) -> None:
    """Verifica que el cierre marque ausente al estudiante sin entrada."""

    service, repository = create_service(
        tmp_path
    )

    session_date = date(
        2026,
        7,
        30,
    )

    service.close_session(
        session_date
    )

    attendance = repository.get_attendance(
        "A002",
        session_date,
    )

    assert attendance is not None
    assert (
        attendance["status"]
        == AttendanceStatus.ABSENT.value
    )
    assert attendance[
        "first_entry_at"
    ] is None
    assert attendance[
        "final_exit_at"
    ] is None
