"""Inicialización del esquema SQLite y sincronización del alumnado."""

from pathlib import Path

from app.database.attendance_repository import (
    AttendanceRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATABASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "attendance.db"
)

STUDENTS_REGISTRY = (
    PROJECT_ROOT
    / "data"
    / "students"
    / "students.json"
)


def main() -> None:
    """Crea el esquema, importa el registro y muestra los alumnos activos."""

    repository = AttendanceRepository(
        DATABASE_PATH
    )

    repository.initialize_schema()

    student_count = (
        repository.sync_students_from_registry(
            STUDENTS_REGISTRY
        )
    )

    print("=" * 60)
    print("BASE DE DATOS INICIALIZADA")
    print("=" * 60)
    print(f"Base de datos: {DATABASE_PATH}")
    print(f"Alumnos:       {student_count}")
    print("=" * 60)

    for student in (
        repository.list_active_students()
    ):
        print(
            f"{student['student_id']} - "
            f"{student['name']}"
        )


if __name__ == "__main__":
    main()
