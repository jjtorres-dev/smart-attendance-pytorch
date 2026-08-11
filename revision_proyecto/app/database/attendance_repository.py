from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any


class AttendanceRepository:
    """
    Acceso a estudiantes, asistencias y eventos mediante SQLite.
    """

    ALLOWED_ATTENDANCE_FIELDS = {
        "status",
        "presence_state",
        "first_entry_at",
        "candidate_exit_at",
        "final_exit_at",
        "updated_at",
    }

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path
        )

        connection.row_factory = sqlite3.Row
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        return connection

    def initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_date TEXT NOT NULL,
                    student_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    presence_state TEXT NOT NULL DEFAULT 'NOT_ENTERED',
                    first_entry_at TEXT,
                    candidate_exit_at TEXT,
                    final_exit_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,

                    FOREIGN KEY (student_id)
                        REFERENCES students(student_id),

                    UNIQUE(session_date, student_id)
                );

                CREATE TABLE IF NOT EXISTS attendance_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_date TEXT NOT NULL,
                    student_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    track_id INTEGER,
                    details TEXT,

                    FOREIGN KEY (student_id)
                        REFERENCES students(student_id)
                );

                CREATE INDEX IF NOT EXISTS
                    idx_attendance_date
                ON attendance(session_date);

                CREATE INDEX IF NOT EXISTS
                    idx_events_student_date
                ON attendance_events(
                    student_id,
                    session_date
                );
                """
            )

    def upsert_student(
        self,
        student_id: str,
        name: str,
        active: bool = True,
    ) -> None:
        now = datetime.now().isoformat(
            timespec="seconds"
        )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO students (
                    student_id,
                    name,
                    active,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)

                ON CONFLICT(student_id)
                DO UPDATE SET
                    name = excluded.name,
                    active = excluded.active,
                    updated_at = excluded.updated_at
                """,
                (
                    student_id,
                    name,
                    int(active),
                    now,
                    now,
                ),
            )

    def sync_students_from_registry(
        self,
        registry_path: str | Path,
    ) -> int:
        path = Path(registry_path)

        if not path.is_file():
            raise FileNotFoundError(
                f"No se encontró el registro: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        students = data.get("students", [])

        for student in students:
            self.upsert_student(
                student_id=str(
                    student["student_id"]
                ),
                name=str(student["name"]),
                active=True,
            )

        return len(students)

    def list_active_students(
        self,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT student_id, name
                FROM students
                WHERE active = 1
                ORDER BY student_id
                """
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def ensure_attendance(
        self,
        student_id: str,
        session_date: date,
    ) -> None:
        now = datetime.now().isoformat(
            timespec="seconds"
        )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO attendance (
                    session_date,
                    student_id,
                    status,
                    presence_state,
                    created_at,
                    updated_at
                )
                VALUES (
                    ?,
                    ?,
                    'PENDING',
                    'NOT_ENTERED',
                    ?,
                    ?
                )
                """,
                (
                    session_date.isoformat(),
                    student_id,
                    now,
                    now,
                ),
            )

    def get_attendance(
        self,
        student_id: str,
        session_date: date,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM attendance
                WHERE student_id = ?
                  AND session_date = ?
                """,
                (
                    student_id,
                    session_date.isoformat(),
                ),
            ).fetchone()

        return (
            dict(row)
            if row is not None
            else None
        )

    def update_attendance(
        self,
        student_id: str,
        session_date: date,
        **fields: Any,
    ) -> None:
        invalid_fields = (
            set(fields)
            - self.ALLOWED_ATTENDANCE_FIELDS
        )

        if invalid_fields:
            raise ValueError(
                "Campos no permitidos: "
                + ", ".join(
                    sorted(invalid_fields)
                )
            )

        if not fields:
            return

        fields.setdefault(
            "updated_at",
            datetime.now().isoformat(
                timespec="seconds"
            ),
        )

        assignments = ", ".join(
            f"{field_name} = ?"
            for field_name in fields
        )

        values = list(fields.values())

        values.extend(
            [
                student_id,
                session_date.isoformat(),
            ]
        )

        with self._connect() as connection:
            connection.execute(
                f"""
                UPDATE attendance
                SET {assignments}
                WHERE student_id = ?
                  AND session_date = ?
                """,
                values,
            )

    def add_event(
        self,
        student_id: str,
        session_date: date,
        event_type: str,
        occurred_at: datetime,
        track_id: int | None = None,
        details: str | None = None,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO attendance_events (
                    session_date,
                    student_id,
                    event_type,
                    occurred_at,
                    track_id,
                    details
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_date.isoformat(),
                    student_id,
                    event_type,
                    occurred_at.isoformat(
                        timespec="seconds"
                    ),
                    track_id,
                    details,
                ),
            )

    def list_attendance(
        self,
        session_date: date,
    ) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    a.*,
                    s.name
                FROM attendance AS a
                INNER JOIN students AS s
                    ON s.student_id = a.student_id
                WHERE a.session_date = ?
                ORDER BY a.student_id
                """,
                (
                    session_date.isoformat(),
                ),
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]