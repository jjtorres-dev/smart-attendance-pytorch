from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.database.attendance_repository import AttendanceRepository


STATUS_LABELS = {
    "PENDING": "PENDIENTE",
    "ON_TIME": "PUNTUAL",
    "LATE": "TARDANZA",
    "ABSENT": "AUSENTE",
}

PRESENCE_LABELS = {
    "NOT_ENTERED": "NO INGRESÓ",
    "INSIDE": "DENTRO",
    "OUTSIDE_TEMPORARY": "FUERA TEMPORALMENTE",
    "CLOSED": "FINALIZADO",
}


@dataclass(frozen=True)
class AttendanceReportRow:
    student_id: str
    student_name: str
    session_date: str
    entry_time: str
    status: str
    exit_time: str
    presence: str


def format_time(
    value: str | None,
) -> str:
    """
    Convierte una fecha ISO a HH:MM:SS.
    """

    if not value:
        return "--"

    try:
        parsed = datetime.fromisoformat(value)

        return parsed.strftime("%H:%M:%S")

    except ValueError:
        return value


def ensure_daily_rows(
    repository: AttendanceRepository,
    session_date: date,
) -> None:
    """
    Garantiza que todos los alumnos activos aparezcan
    en la lista de asistencia aunque todavía no hayan ingresado.
    """

    for student in repository.list_active_students():
        repository.ensure_attendance(
            student_id=student["student_id"],
            session_date=session_date,
        )


def build_report_rows(
    repository: AttendanceRepository,
    session_date: date,
) -> list[AttendanceReportRow]:
    ensure_daily_rows(
        repository=repository,
        session_date=session_date,
    )

    attendance_rows = repository.list_attendance(
        session_date
    )

    report_rows: list[AttendanceReportRow] = []

    for attendance in attendance_rows:
        status_code = attendance["status"]
        presence_code = attendance["presence_state"]

        first_entry = attendance.get(
            "first_entry_at"
        )

        final_exit = attendance.get(
            "final_exit_at"
        )

        candidate_exit = attendance.get(
            "candidate_exit_at"
        )

        # Si todavía no se ha cerrado la sesión,
        # mostramos la salida provisional cuando corresponda.
        if final_exit:
            exit_time = format_time(final_exit)

        elif (
            presence_code == "OUTSIDE_TEMPORARY"
            and candidate_exit
        ):
            exit_time = (
                f"{format_time(candidate_exit)}*"
            )

        else:
            exit_time = "--"

        report_rows.append(
            AttendanceReportRow(
                student_id=str(
                    attendance["student_id"]
                ),
                student_name=str(
                    attendance["name"]
                ),
                session_date=(
                    session_date.isoformat()
                ),
                entry_time=format_time(
                    first_entry
                ),
                status=STATUS_LABELS.get(
                    status_code,
                    status_code,
                ),
                exit_time=exit_time,
                presence=PRESENCE_LABELS.get(
                    presence_code,
                    presence_code,
                ),
            )
        )

    return report_rows


def calculate_summary(
    rows: list[AttendanceReportRow],
) -> dict[str, int]:
    summary = {
        "TOTAL": len(rows),
        "PUNTUAL": 0,
        "TARDANZA": 0,
        "AUSENTE": 0,
        "PENDIENTE": 0,
    }

    for row in rows:
        if row.status in summary:
            summary[row.status] += 1

    return summary


def print_report(
    rows: list[AttendanceReportRow],
    session_date: date,
) -> None:
    """
    Imprime una tabla legible en consola.
    """

    summary = calculate_summary(rows)

    print()
    print("=" * 115)
    print(
        f"LISTA DE ASISTENCIA - "
        f"{session_date.strftime('%d/%m/%Y')}"
    )
    print("=" * 115)

    header = (
        f"{'CÓDIGO':<10}"
        f"{'ALUMNO':<38}"
        f"{'ENTRADA':<12}"
        f"{'ESTADO':<14}"
        f"{'SALIDA':<13}"
        f"{'SITUACIÓN':<24}"
    )

    print(header)
    print("-" * 115)

    for row in rows:
        print(
            f"{row.student_id:<10}"
            f"{row.student_name[:36]:<38}"
            f"{row.entry_time:<12}"
            f"{row.status:<14}"
            f"{row.exit_time:<13}"
            f"{row.presence:<24}"
        )

    print("-" * 115)

    print(
        f"TOTAL: {summary['TOTAL']} | "
        f"PUNTUALES: {summary['PUNTUAL']} | "
        f"TARDANZAS: {summary['TARDANZA']} | "
        f"AUSENTES: {summary['AUSENTE']} | "
        f"PENDIENTES: {summary['PENDIENTE']}"
    )

    print("=" * 115)

    if any(
        row.exit_time.endswith("*")
        for row in rows
    ):
        print(
            "* La hora de salida marcada con * "
            "es provisional porque la sesión todavía no ha cerrado."
        )

    print()


def export_report_csv(
    rows: list[AttendanceReportRow],
    session_date: date,
    output_directory: str | Path,
) -> Path:
    """
    Exporta la asistencia en CSV UTF-8 compatible con Excel.
    """

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_directory
        / (
            f"asistencia_"
            f"{session_date.isoformat()}.csv"
        )
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.writer(
            file,
            delimiter=";",
        )

        writer.writerow(
            [
                "Código",
                "Alumno",
                "Fecha",
                "Hora de entrada",
                "Estado",
                "Hora de salida",
                "Situación",
            ]
        )

        for row in rows:
            # Quitamos el asterisco del CSV.
            exit_time = row.exit_time.rstrip("*")

            writer.writerow(
                [
                    row.student_id,
                    row.student_name,
                    row.session_date,
                    (
                        ""
                        if row.entry_time == "--"
                        else row.entry_time
                    ),
                    row.status,
                    (
                        ""
                        if exit_time == "--"
                        else exit_time
                    ),
                    row.presence,
                ]
            )

    return output_path