"""
database.py
-----------
All SQL lives here. Uses sqlite3, wrapped in a Database class so nothing
outside this file ever writes raw SQL.

Per the assignment requirement: every sqlite3.connect() uses a `with`
block, since Connection supports the context manager protocol (it commits
or rolls back automatically on exit). sqlite3.Cursor does NOT support
`with`, so cursor objects returned by conn.execute() are used directly
without wrapping them in a `with` block - wrapping them would raise
AttributeError, since Cursor has no __enter__/__exit__.
"""

import hashlib
import os
import sqlite3
from datetime import date, datetime

from models import Student, User


def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


class Database:
    def __init__(self, db_path: str = "attendance.db"):
        self._db_path = db_path
        self._create_tables()
        self._ensure_default_admin()

    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _create_tables(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT NOT NULL,
                    roll_no     TEXT NOT NULL UNIQUE,
                    department  TEXT NOT NULL,
                    semester    INTEGER NOT NULL,
                    is_active   INTEGER NOT NULL DEFAULT 1
                )
            """)
            # Older databases created before is_active existed won't have the
            # column - add it if it's missing, so upgrading in place doesn't
            # break on an existing attendance.db file.
            existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(students)")]
            if "is_active" not in existing_cols:
                conn.execute("ALTER TABLE students ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id  INTEGER NOT NULL,
                    date        TEXT NOT NULL,
                    time        TEXT NOT NULL,
                    FOREIGN KEY (student_id) REFERENCES students(id),
                    UNIQUE(student_id, date)
                )
            """)   
            att_cols = [row[1] for row in conn.execute("PRAGMA table_info(attendance)")]
            if "status" not in att_cols:
                conn.execute("ALTER TABLE attendance ADD COLUMN status TEXT NOT NULL DEFAULT 'On time'")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    username      TEXT NOT NULL UNIQUE,
                    salt          TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role          TEXT NOT NULL CHECK(role IN ('admin','teacher','student')),
                    student_id    INTEGER,
                    FOREIGN KEY (student_id) REFERENCES students(id)
                )
            """)

    def _ensure_default_admin(self):
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM users WHERE role = 'admin'").fetchone()
            if row is None:
                self.add_user("admin", "admin123", "admin")

    # ---- Users / login ----
    def add_user(self, username: str, password: str, role: str, student_id: int = None) -> int:
        salt = os.urandom(8).hex()
        pw_hash = _hash_password(password, salt)
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, salt, password_hash, role, student_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (username, salt, pw_hash, role, student_id),
            )
            return cur.lastrowid

    def verify_login(self, username: str, password: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, salt, password_hash, role, student_id FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        if row is None:
            return None
        user_id, salt, stored_hash, role, student_id = row
        if _hash_password(password, salt) != stored_hash:
            return None
        return User(username=username, role=role, user_id=user_id, student_id=student_id)

    def list_users(self) -> list[dict]:
        """Returns every login account, with the linked student's name shown
        where applicable - lets an admin see what accounts already exist
        before creating a new one (e.g. to avoid duplicate usernames)."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT u.username, u.role, s.name, s.roll_no
                FROM users u LEFT JOIN students s ON s.id = u.student_id
                ORDER BY u.role, u.username
            """).fetchall()
        return [{"username": r[0], "role": r[1], "student_name": r[2], "student_roll": r[3]}
                for r in rows]

    # ---- Student CRUD ----
    def add_student(self, student: Student) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO students (name, roll_no, department, semester) VALUES (?, ?, ?, ?)",
                (student.name, student.roll_no, student.department, student.semester),
            )
            return cur.lastrowid

    def _row_to_student(self, r) -> Student:
        return Student(name=r[1], roll_no=r[2], department=r[3], semester=r[4],
                        student_id=r[0], is_active=bool(r[5]))

    def get_all_students(self, include_inactive: bool = False) -> list[Student]:
        query = "SELECT id, name, roll_no, department, semester, is_active FROM students"
        if not include_inactive:
            query += " WHERE is_active = 1"
        query += " ORDER BY name"
        with self._connect() as conn:
            rows = conn.execute(query).fetchall()
        return [self._row_to_student(r) for r in rows]

    def search_students(self, query: str, include_inactive: bool = False) -> list[Student]:
        """Searches by name or roll number (partial, case-insensitive)."""
        like = f"%{query}%"
        sql = ("SELECT id, name, roll_no, department, semester, is_active FROM students "
               "WHERE (name LIKE ? OR roll_no LIKE ?)")
        if not include_inactive:
            sql += " AND is_active = 1"
        sql += " ORDER BY name"
        with self._connect() as conn:
            rows = conn.execute(sql, (like, like)).fetchall()
        return [self._row_to_student(r) for r in rows]

    def get_student_by_id(self, student_id: int) -> Student | None:
        with self._connect() as conn:
            r = conn.execute(
                "SELECT id, name, roll_no, department, semester, is_active FROM students WHERE id = ?",
                (student_id,),
            ).fetchone()
        if r is None:
            return None
        return self._row_to_student(r)

    def update_student(self, student_id: int, name: str, roll_no: str,
                        department: str, semester: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE students SET name = ?, roll_no = ?, department = ?, semester = ? WHERE id = ?",
                (name, roll_no, department, semester, student_id),
            )
            return cur.rowcount > 0

    def set_student_active(self, student_id: int, active: bool) -> bool:
        """Deactivating a student hides them from the default roster and
        stops them from being marked present, WITHOUT deleting their
        student record, attendance history, or saved face images. This is
        deliberately reversible - see set_student_active(student_id, True)
        to bring them back."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE students SET is_active = ? WHERE id = ?",
                (1 if active else 0, student_id),
            )
            return cur.rowcount > 0

    # ---- Settings (late cutoff) ----
    def get_late_time(self) -> str:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM settings WHERE key = 'late_time'").fetchone()
        return row[0] if row else "09:00"

    def set_late_time(self, hhmm: str) -> None:
        hhmm = datetime.strptime(hhmm, "%H:%M").strftime("%H:%M")  # ValueError if malformed
        with self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('late_time', ?)", (hhmm,))

    # ---- Attendance ----
    def mark_attendance(self, student_id: int) -> bool:
        """Inserts today's attendance row for this student. Returns True if
        this call actually created the row, False if one already existed
        (caught via the UNIQUE(student_id, date) constraint - this check
        is atomic at the database level, so it stays correct even if two
        clients try to mark the same student at the exact same instant)."""
        today = date.today().isoformat()
        now = datetime.now().strftime("%H:%M")
        status = "On time" if now <= self.get_late_time() else "Late"
        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, ?)",
                    (student_id, today, now, status),
                )
            return True
        except sqlite3.IntegrityError:
            return False  # already marked today

    def get_attendance_for_date(self, day: str = None) -> list[tuple]:
        day = day or date.today().isoformat()
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT s.name, s.roll_no, a.time, a.status
                FROM attendance a JOIN students s ON s.id = a.student_id
                WHERE a.date = ? ORDER BY a.time
            """, (day,)).fetchall()
        return rows

    def get_all_attendance(self) -> list[tuple]:
        """Every attendance record ever recorded, most recent first. For a
        single classroom over one semester this is a small enough result
        set to just return in full - no pagination needed at this scale."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT s.name, s.roll_no, a.date, a.time, a.status
                FROM attendance a JOIN students s ON s.id = a.student_id
                ORDER BY a.date DESC, a.time DESC
            """).fetchall()
        return rows

    def search_attendance(self, query: str) -> list[tuple]:
        """Searches attendance history by student name or roll number
        (partial, case-insensitive), across every date on record."""
        like = f"%{query}%"
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT s.name, s.roll_no, a.date, a.time, a.status
                FROM attendance a JOIN students s ON s.id = a.student_id
                WHERE s.name LIKE ? OR s.roll_no LIKE ?
                ORDER BY a.date DESC, a.time DESC
            """, (like, like)).fetchall()
        return rows

    def get_attendance_for_student(self, student_id: int, limit: int = 30) -> list[tuple]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT date, time, status FROM attendance
                WHERE student_id = ? ORDER BY date DESC LIMIT ?
            """, (student_id, limit)).fetchall()
        return rows
