"""
server.py
---------
One central server. Every client (Admin, Teacher, Student) connects here
over a TCP socket. A client's role is only known after it logs in; every
request after that is checked against PERMISSIONS before it's allowed to
run.

Handles multiple clients at once: one thread per connection. The database
is safe under this because every Database method opens its own
short-lived sqlite3 connection per call. The FaceRecognizer is different:
it's ONE shared object across every thread, and OpenCV's recognizer isn't
safe to call from two threads at the same instant - recognizer_lock
serializes access to it.

_marked_today cache: an in-memory set of student IDs already marked
present today, rebuilt from scratch whenever the date rolls over. This
exists purely for efficiency, NOT correctness - the database's
UNIQUE(student_id, date) constraint is what actually guarantees no
duplicate row can ever be created, even if this cache were somehow wrong
or two threads raced each other. What the cache saves is a wasted INSERT
attempt (and the exception-handling that catches it) every time a face
that's already been marked lingers in front of a splash screen's camera -
common, since someone doesn't vanish the instant they're marked present.

Run this first, then run client_gui.py (as many times as you like, from
any machine on the network that can reach this one).
"""

import socket
import threading
from datetime import date

from database import Database
from face_recognizer import FaceRecognizer
from models import Student
from protocol import recv_msg, send_msg

HOST = "0.0.0.0"
PORT = 5050

db = Database()
recognizer = FaceRecognizer()
recognizer_lock = threading.Lock()  # serializes access to the shared recognizer

_marked_today: set[int] = set()
_marked_today_date = date.today()
_marked_today_lock = threading.Lock()

PERMISSIONS = {
    "public": {"recognize_frame"},
    "admin": {"register_student", "add_face_sample", "train_model",
              "list_students", "search_students", "update_student",
              "set_student_active", "list_attendance_today",
              "list_attendance_for_date", "list_attendance_all",
              "search_attendance", "mark_attendance_manual",
              "create_user", "list_users", "clear_face_samples",
              "get_late_time", "set_late_time"},
    "teacher": {"list_students", "search_students", "list_attendance_today",
                "list_attendance_for_date", "list_attendance_all",
                "search_attendance", "mark_attendance_manual"},
    "student": {"my_attendance"},
}


def permitted(role, action):
    if action in PERMISSIONS["public"]:
        return True
    return action in PERMISSIONS.get(role, set())


def _mark_attendance_cached(student_id: int) -> bool:
    """Wraps db.mark_attendance() with the in-memory cache described
    above. Returns True only if this call is what actually created
    today's attendance row for this student."""
    global _marked_today_date
    with _marked_today_lock:
        today = date.today()
        if today != _marked_today_date:
            _marked_today.clear()          # new day, start fresh
            _marked_today_date = today
        if student_id in _marked_today:
            return False                   # already marked - skip the DB call entirely

    marked = db.mark_attendance(student_id)
    if marked:
        with _marked_today_lock:
            _marked_today.add(student_id)
    else:
        # Cache said "not marked yet" but the DB constraint disagreed -
        # e.g. another thread marked this student between the cache
        # check above and this INSERT. The DB is the authority; bring
        # the cache back in sync with it rather than trusting our guess.
        with _marked_today_lock:
            _marked_today.add(student_id)
    return marked


def handle_action(action, payload, session):
    """Dispatches one request to the right piece of logic. `session` holds
    the logged-in user for this connection (or None before login)."""

    if action == "login":
        user = db.verify_login(payload["username"], payload["password"])
        if user is None:
            return {"ok": False, "error": "invalid username or password"}
        session["user"] = user
        return {"ok": True, "role": user.role, "username": user.username,
                "student_id": user.student_id}

    if action == "recognize_frame":
        # The client already detected a face locally and cropped to it -
        # see face_detect.py and client_gui.py. No detection happens here.
        with recognizer_lock:
            student_id, confidence = recognizer.recognize_crop(payload["image"])
        if student_id is None:
            return {"ok": True, "match": False}
        student = db.get_student_by_id(student_id)
        if student is None or not student.is_active:
            # Either the ID doesn't exist, or the student has been
            # deactivated - either way, treat it exactly like "no match."
            # This is what actually stops a deactivated student from
            # continuing to be marked present: their face images and
            # trained model entry still exist (deactivating never deletes
            # them), but recognition results for them are ignored here.
            return {"ok": True, "match": False}
        marked = _mark_attendance_cached(student_id)
        return {"ok": True, "match": True, "name": student.name,
                "roll_no": student.roll_no, "marked": marked}

    if action == "logout":
        session["user"] = None
        return {"ok": True}
    
    # everything below requires login
    user = session.get("user")
    role = user.role if user else None
    if not permitted(role, action):
        return {"ok": False, "error": "not authorized for this action"}

    if action == "register_student":
        student = Student(payload["name"], payload["roll_no"],
                           payload["department"], int(payload["semester"]))
        student_id = db.add_student(student)
        return {"ok": True, "student_id": student_id}

    if action == "add_face_sample":
        with recognizer_lock:
            count = recognizer.save_sample(payload["student_id"], payload["image"])
        if count == -1:
            return {"ok": True, "face_found": False}
        return {"ok": True, "face_found": True, "sample_count": count}

    if action == "clear_face_samples":
        with recognizer_lock:
            removed = recognizer.clear_samples(payload["student_id"])
        return {"ok": True, "removed": removed}

    if action == "get_late_time":
        return {"ok": True, "late_time": db.get_late_time()}

    if action == "set_late_time":
        try:
            db.set_late_time(payload["late_time"])
        except ValueError:
            return {"ok": False, "error": "invalid time - use HH:MM (24-hour), e.g. 09:30"}
        return {"ok": True}
    
    if action == "train_model":
        try:
            with recognizer_lock:
                image_count, student_count = recognizer.train()
            return {"ok": True, "image_count": image_count, "student_count": student_count}
        except ValueError as e:
            return {"ok": False, "error": str(e)}

    if action == "list_students":
        include_inactive = bool(payload.get("include_inactive", False))
        students = db.get_all_students(include_inactive=include_inactive)
        return {"ok": True, "students": [
            {"id": s.student_id, "name": s.name, "roll_no": s.roll_no,
             "department": s.department, "semester": s.semester,
             "is_active": s.is_active} for s in students
        ]}

    if action == "search_students":
        include_inactive = bool(payload.get("include_inactive", False))
        students = db.search_students(payload.get("query", ""), include_inactive=include_inactive)
        return {"ok": True, "students": [
            {"id": s.student_id, "name": s.name, "roll_no": s.roll_no,
             "department": s.department, "semester": s.semester,
             "is_active": s.is_active} for s in students
        ]}

    if action == "update_student":
        ok = db.update_student(payload["student_id"], payload["name"], payload["roll_no"],
                                payload["department"], int(payload["semester"]))
        return {"ok": ok} if ok else {"ok": False, "error": "student not found"}

    if action == "set_student_active":
        ok = db.set_student_active(payload["student_id"], bool(payload["active"]))
        return {"ok": ok} if ok else {"ok": False, "error": "student not found"}

    if action == "list_attendance_today":
        rows = db.get_attendance_for_date()
        return {"ok": True, "records": [{"name": n, "roll_no": r, "time": t, "status": s} for n, r, t, s in rows]}

    if action == "list_attendance_for_date":
        day = payload.get("date")  # expected "YYYY-MM-DD"; defaults to today if omitted
        try:
            rows = db.get_attendance_for_date(day)
        except Exception:
            return {"ok": False, "error": "invalid date format - use YYYY-MM-DD"}
        return {"ok": True, "records": [{"name": n, "roll_no": r, "time": t, "status": s} for n, r, t, s in rows]}

    if action == "list_attendance_all":
        rows = db.get_all_attendance()
        return {"ok": True, "records": [
            {"name": n, "roll_no": r, "date": d, "time": t, "status": s} for n, r, d, t, s in rows
        ]}

    if action == "search_attendance":
        rows = db.search_attendance(payload.get("query", ""))
        return {"ok": True, "records": [
            {"name": n, "roll_no": r, "date": d, "time": t, "status": s} for n, r, d, t, s in rows
        ]}

    if action == "mark_attendance_manual":
        student = db.get_student_by_id(payload["student_id"])
        if student is None or not student.is_active:
            return {"ok": False, "error": "student not found or inactive"}
        marked = _mark_attendance_cached(student.student_id)
        return {"ok": True, "marked": marked, "name": student.name, "roll_no": student.roll_no}

    if action == "create_user":
        try:
            user_id = db.add_user(payload["username"], payload["password"],
                                   payload["role"], payload.get("student_id"))
        except Exception as e:
            return {"ok": False, "error": f"could not create account: {e}"}
        return {"ok": True, "user_id": user_id}

    if action == "list_users":
        return {"ok": True, "users": db.list_users()}

    if action == "my_attendance":
        if user.student_id is None:
            return {"ok": False, "error": "this account isn't linked to a student record"}
        rows = db.get_attendance_for_student(user.student_id)
        return {"ok": True, "records": [{"date": d, "time": t, "status": s} for d, t, s in rows]}

    return {"ok": False, "error": f"unknown action: {action}"}


def handle_client(conn: socket.socket, addr):
    print(f"[server] connected: {addr}")
    session = {"user": None}
    try:
        with conn:
            while True:
                try:
                    request = recv_msg(conn)
                except ConnectionError:
                    break
                action = request.get("action")
                payload = request.get("payload", {})
                try:
                    response = handle_action(action, payload, session)
                except Exception as e:
                    response = {"ok": False, "error": f"server error: {e}"}
                send_msg(conn, response)
    finally:
        print(f"[server] disconnected: {addr}")


def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen()
        print(f"[server] listening on {HOST}:{PORT}")
        while True:
            conn, addr = server_sock.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            thread.start()


if __name__ == "__main__":
    main()
