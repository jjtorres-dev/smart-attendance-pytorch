"""Interfaz de consola para consultar, cerrar y exportar asistencias."""

from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from app.attendance.report import (
    build_report_rows,
    export_report_csv,
    print_report,
)
from app.attendance.service import (
    AttendancePolicy,
    AttendanceService,
)
from app.database.attendance_repository import (
    AttendanceRepository,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATABASE = (
    PROJECT_ROOT
    / "data"
    / "attendance.db"
)

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "config"
    / "attendance.json"
)

DEFAULT_REGISTRY = (
    PROJECT_ROOT
    / "data"
    / "students"
    / "students.json"
)

DEFAULT_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "attendance"
)


def parse_date(
    value: str,
) -> date:
    """Convierte una fecha ISO recibida por CLI y valida su formato."""

    try:
        return date.fromisoformat(value)

    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "La fecha debe usar formato YYYY-MM-DD."
        ) from error


def run(
    args: argparse.Namespace,
) -> None:
    """Ejecuta la consulta, el cierre opcional y la exportación del reporte."""

    database_path = args.database.resolve()
    config_path = args.attendance_config.resolve()
    registry_path = args.registry.resolve()
    output_directory = (
        args.output_directory.resolve()
    )

    repository = AttendanceRepository(
        database_path
    )

    repository.initialize_schema()

    repository.sync_students_from_registry(
        registry_path
    )

    policy = AttendancePolicy.from_json(
        config_path
    )

    service = AttendanceService(
        repository=repository,
        policy=policy,
    )

    timezone = ZoneInfo(
        policy.timezone_name
    )

    current_local_datetime = datetime.now(
        timezone
    )

    session_date = (
        args.date
        if args.date is not None
        else current_local_datetime.date()
    )

    if args.close_session:
        session_end_datetime = datetime.combine(
            session_date,
            policy.session_end,
            tzinfo=timezone,
        )

        can_close_normally = (
            current_local_datetime
            >= session_end_datetime
        )

        # El cierre anticipado solo se acepta cuando el usuario lo declara
        # explícitamente con la opción reservada para pruebas controladas.
        if (
            not can_close_normally
            and not args.force_close
        ):
            raise RuntimeError(
                "La sesión todavía no ha terminado. "
                f"Hora de cierre configurada: "
                f"{policy.session_end}. "
                "Para una prueba controlada puedes "
                "utilizar --force-close."
            )

        summary = service.close_session(
            session_date
        )

        print()
        print("SESIÓN CERRADA")
        print(
            f"Total: {summary['TOTAL']} | "
            f"Puntuales: {summary['ON_TIME']} | "
            f"Tardanzas: {summary['LATE']} | "
            f"Ausentes: {summary['ABSENT']}"
        )

    rows = build_report_rows(
        repository=repository,
        session_date=session_date,
    )

    print_report(
        rows=rows,
        session_date=session_date,
    )

    if args.export:
        report_path = export_report_csv(
            rows=rows,
            session_date=session_date,
            output_directory=output_directory,
        )

        print(
            f"Reporte CSV generado:\n"
            f"{report_path}"
        )


def build_parser() -> argparse.ArgumentParser:
    """Construye el analizador de argumentos del reporte de asistencia."""

    parser = argparse.ArgumentParser(
        description=(
            "Muestra y exporta la lista de asistencia."
        )
    )

    parser.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help="Base de datos SQLite.",
    )

    parser.add_argument(
        "--attendance-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Configuración del horario.",
    )

    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Registro de alumnos.",
    )

    parser.add_argument(
        "--date",
        type=parse_date,
        default=None,
        help="Fecha YYYY-MM-DD. Por defecto usa hoy.",
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_REPORT_DIRECTORY,
    )

    parser.add_argument(
        "--export",
        action="store_true",
        help="Genera también un archivo CSV.",
    )

    parser.add_argument(
        "--close-session",
        action="store_true",
        help=(
            "Cierra la sesión antes de generar "
            "el reporte."
        ),
    )

    parser.add_argument(
        "--force-close",
        action="store_true",
        help=(
            "Permite cerrar manualmente una sesión "
            "antes del horario. Solo para pruebas."
        ),
    )

    return parser


def main() -> None:
    """Valida los argumentos de consola e inicia la generación del reporte."""

    parser = build_parser()
    arguments = parser.parse_args()

    if (
        arguments.force_close
        and not arguments.close_session
    ):
        parser.error(
            "--force-close requiere --close-session."
        )

    run(arguments)


if __name__ == "__main__":
    main()
