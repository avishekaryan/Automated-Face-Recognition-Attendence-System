"""
client_gui.py
-------------
Runs on any machine that can reach the server. Flow:

    Splash screen (camera auto-recognizes + logs attendance, no login needed)
                    |
              [ Login button ]
                    |
        Login screen (pick Admin / Teacher / Student, enter credentials)
                    |
      Role dashboard (server enforces what each role can actually do)

What the client does and does NOT decide:
    - It DOES run local face DETECTION (face_detect.py) on the splash
      screen, purely to answer "is anyone in frame right now" and, if so,
      "roughly where." This is why an empty room stops sending anything
      to the server at all, and why what IS sent is already cropped down
      to the face region instead of the whole frame.
    - It does NOT decide WHOSE face it is, and it does NOT decide whether
      to mark attendance. Both of those only ever happen on the server,
      which is what makes it safe against a tampered client - see the
      longer note in SplashFrame below.
"""

import base64
import csv
import time
import socket
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import cv2
from PIL import Image, ImageTk

from face_detect import detect_faces
from protocol import recv_msg, send_msg

HOST = "127.0.0.1"  # change to the server's LAN IP when running on another machine
PORT = 5050

GREEN = (0, 200, 0)   # BGR - attendance is on record
RED = (0, 0, 220)     # BGR - a face is visible but not recognized

# (instruction shown to the person, how many samples to take for that pose)
CAPTURE_STEPS = [
    ("Look STRAIGHT at the camera", 8),
    ("Turn your head slightly to the LEFT", 5),
    ("Turn your head slightly to the RIGHT", 5),
    ("Tilt your chin slightly UP", 4),
    ("Tilt your chin slightly DOWN", 4),
    ("Look straight again and SMILE", 4),
]

class ServerConnection:
    def __init__(self, host, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))

    def call(self, action, payload=None):
        send_msg(self.sock, {"action": action, "payload": payload or {}})
        return recv_msg(self.sock)


def frame_to_b64(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf.tobytes()).decode("ascii")

def check_capture_quality(frame):
    """Returns (problem, box). problem is None when the frame is good enough
    to save, otherwise a short instruction for the person. box is the face
    box found locally, or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()
    if brightness < 70:
        return "Too dark - face a light or turn the room light on", None
    if brightness > 200:
        return "Too bright - avoid a strong light/window behind or in front", None
    faces = detect_faces(gray)
    if len(faces) == 0:
        return "No face found - face the camera and stay in view", None
    x, y, w, h = faces[0]
    if w < 120:
        return "Move closer to the camera", (x, y, w, h)
    if w > 380:
        return "Move a little farther from the camera", (x, y, w, h)
    if cv2.Laplacian(gray[y:y + h, x:x + w], cv2.CV_64F).var() < 40:
        return "Image is blurry - hold still", (x, y, w, h)
    return None, (x, y, w, h)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Recognition Attendance System")
        self.geometry("1000x680")
        self.minsize(900, 620)

        try:
            self.conn = ServerConnection(HOST, PORT)
        except ConnectionRefusedError:
            messagebox.showerror("Can't connect", f"No server found at {HOST}:{PORT}.\n"
                                  "Start server.py first.")
            self.destroy()
            return

        self.user = None  # set after a successful login

        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)
        self.show(SplashFrame)

    def show(self, frame_class, **kwargs):
        for child in self.container.winfo_children():
            child.destroy()
        frame = frame_class(self.container, self, **kwargs)
        frame.pack(fill="both", expand=True)

    def logout(self):
        self.conn.call("logout")   # also ends the login on the server side
        self.user = None
        self.show(SplashFrame)


class SplashFrame(tk.Frame):
    """No login required: the camera runs continuously, detects faces
    locally, and asks the server to identify anyone it finds."""

    def __init__(self, parent, app: App):
        super().__init__(parent)
        self.app = app
        self.cap = cv2.VideoCapture(0)

        tk.Label(self, text="Live Attendance", font=("Segoe UI", 16)).pack(pady=10)
        self.video_label = tk.Label(self, bg="black")
        self.video_label.pack(pady=5)
        self.status_var = tk.StringVar(value="Watching for a face...")
        tk.Label(self, textvariable=self.status_var, font=("Segoe UI", 12)).pack(pady=8)
        ttk.Button(self, text="Login", command=self.go_login).pack(pady=15)

        self._last_frame = None
        self._box = None        # (x, y, w, h) of the most recently detected face, or None
        self._box_color = None  # GREEN, RED, or None (no box drawn)
        self._update_preview()
        self._recognize_loop()

    def _update_preview(self):
        ok, frame = self.cap.read()
        if ok:
            self._last_frame = frame
            display = frame.copy()
            if self._box is not None and self._box_color is not None:
                x, y, w, h = self._box
                cv2.rectangle(display, (x, y), (x + w, y + h), self._box_color, 2)
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((420, 315))
            photo = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=photo)
            self.video_label.image = photo
        self.after(30, self._update_preview)

    def _recognize_loop(self):
        # Runs about once a second. Detection happens HERE, locally, first -
        # if nobody's in frame, nothing is sent to the server at all.
        if self._last_frame is not None:
            gray = cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2GRAY)
            faces = detect_faces(gray)

            if len(faces) == 0:
                self._box = None
                self._box_color = None
                self.status_var.set("Watching for a face...")
            else:
                x, y, w, h = faces[0]
                self._box = (x, y, w, h)
                # crop with a small margin so the server has a little context,
                # clamped so it never reads outside the frame
                pad = 0
                y0, y1 = max(0, y - pad), y + h + pad
                x0, x1 = max(0, x - pad), x + w + pad
                crop = self._last_frame[y0:y1, x0:x1]

                try:
                    result = self.app.conn.call("recognize_frame", {"image": frame_to_b64(crop)})
                except (ConnectionError, OSError):
                    self.status_var.set("Lost connection to server.")
                    self._box_color = RED
                    self.after(1000, self._recognize_loop)
                    return

                if result.get("match"):
                    self._box_color = GREEN
                    note = "marked present" if result["marked"] else "already marked today"
                    self.status_var.set(f"{result['name']} ({result['roll_no']}) - {note}")
                else:
                    self._box_color = RED
                    self.status_var.set("Face detected, not recognized")

        self.after(1000, self._recognize_loop)

    def go_login(self):
        self.cap.release()
        self.app.show(LoginFrame)


class LoginFrame(tk.Frame):
    def __init__(self, parent, app: App):
        super().__init__(parent, bg="#eef2f7")
        self.app = app
        self.role_var = tk.StringVar(value="admin")
        self.username_var = tk.StringVar()
        self.password_var = tk.StringVar()

        card = tk.Frame(self, bg="white", highlightbackground="#d0d7e2", highlightthickness=1)
        card.place(relx=0.5, rely=0.5, anchor="center")   # stays centered at any window size
        inner = tk.Frame(card, bg="white")
        inner.pack(padx=50, pady=36)

        tk.Label(inner, text="Welcome back", font=("Segoe UI", 24, "bold"),
                 bg="white", fg="#1f2937").pack()
        tk.Label(inner, text="Sign in to the Attendance System", font=("Segoe UI", 11),
                 bg="white", fg="#6b7280").pack(pady=(2, 20))

        role_bar = tk.Frame(inner, bg="white")
        role_bar.pack(pady=(0, 18))
        self.role_buttons = {}
        for role in ("admin", "teacher", "student"):
            b = tk.Radiobutton(role_bar, text=role.capitalize(), variable=self.role_var, value=role,
                               indicatoron=False, font=("Segoe UI", 11, "bold"), width=10, pady=6,
                               bd=0, relief="flat", cursor="hand2", selectcolor="#2563eb",
                               activebackground="#e5e7eb")
            b.pack(side="left", padx=3)
            self.role_buttons[role] = b
        self.role_var.trace_add("write", lambda *_: self._style_roles())
        self._style_roles()

        tk.Label(inner, text="Username", font=("Segoe UI", 11), bg="white",
                 fg="#374151", anchor="w").pack(fill="x")
        self.username_entry = tk.Entry(inner, textvariable=self.username_var, font=("Segoe UI", 14),
                                       width=28, relief="solid", bd=1)
        self.username_entry.pack(fill="x", ipady=6, pady=(2, 12))

        tk.Label(inner, text="Password", font=("Segoe UI", 11), bg="white",
                 fg="#374151", anchor="w").pack(fill="x")
        self.password_entry = tk.Entry(inner, textvariable=self.password_var, show="*",
                                       font=("Segoe UI", 14), width=28, relief="solid", bd=1)
        self.password_entry.pack(fill="x", ipady=6, pady=(2, 6))

        tk.Checkbutton(inner, text="Show password", bg="white", activebackground="white",
                       font=("Segoe UI", 10), anchor="w",
                       command=lambda: self.password_entry.configure(
                           show="" if self.password_entry.cget("show") else "*")).pack(anchor="w")

        tk.Button(inner, text="Sign in", command=self._sign_in, font=("Segoe UI", 13, "bold"),
                  bg="#2563eb", fg="white", activebackground="#1d4ed8", activeforeground="white",
                  relief="flat", cursor="hand2", pady=8).pack(fill="x", pady=(18, 8))
        tk.Button(inner, text="← Back", command=lambda: self.app.show(SplashFrame),
                  font=("Segoe UI", 11), bg="white", fg="#4b5563", relief="flat",
                  cursor="hand2", pady=4).pack(fill="x")

        self.username_entry.bind("<Return>", lambda e: self.password_entry.focus_set())
        self.password_entry.bind("<Return>", lambda e: self._sign_in())
        self.after(100, self.username_entry.focus_set)

    def _style_roles(self):
        for role, b in self.role_buttons.items():
            selected = self.role_var.get() == role
            b.configure(bg="#2563eb" if selected else "#f3f4f6",
                        fg="white" if selected else "#374151")
    def _sign_in(self):
        result = self.app.conn.call("login", {
            "username": self.username_var.get(), "password": self.password_var.get(),
        })
        if not result.get("ok"):
            messagebox.showerror("Login failed", result.get("error", "Unknown error"))
            return
        if result["role"] != self.role_var.get():
            messagebox.showerror("Wrong role", f"This account is a {result['role']} account.")
            return
        self.app.user = result
        dashboards = {"admin": AdminFrame, "teacher": TeacherFrame, "student": StudentFrame}
        self.app.show(dashboards[result["role"]])


class AdminFrame(tk.Frame):
    """Admin dashboard, organized into tabs so each concern (an overview,
    registering, managing the roster, viewing records, managing accounts)
    has its own space instead of one crowded screen."""

    def __init__(self, parent, app: App):
        super().__init__(parent)
        self.app = app

        tk.Label(self, text=f"Admin — {app.user['username']}", font=("Segoe UI", 15)).pack(pady=8)
        ttk.Button(self, text="← Logout", command=app.logout).pack(anchor="w", padx=10)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        register_tab = RegisterTab(notebook, app)
        students_tab = ManageStudentsTab(notebook, app)
        attendance_tab = AttendanceTab(notebook, app)
        accounts_tab = ManageAccountsTab(notebook, app)
        dashboard_tab = DashboardTab(notebook, app, notebook)

        # Added in this order so DashboardTab's quick-launch buttons can
        # jump straight to a fixed tab index (1 = Register, 2 = Manage
        # Students, 3 = Attendance, 4 = Accounts) via notebook.select(i).
        notebook.add(dashboard_tab, text="Dashboard")
        notebook.add(register_tab, text="Register Student")
        notebook.add(students_tab, text="Manage Students")
        notebook.add(attendance_tab, text="Attendance Records")
        notebook.add(accounts_tab, text="Manage Accounts")


class DashboardTab(tk.Frame):
    """The landing screen after Admin login: a quick summary plus buttons
    that jump straight to each tool, instead of dropping straight into
    the registration form with no overview first."""

    def __init__(self, parent, app: App, notebook: ttk.Notebook):
        super().__init__(parent)
        self.app = app
        self.notebook = notebook

        tk.Label(self, text="Overview", font=("Segoe UI", 14, "bold")).pack(pady=(15, 5))

        stats = tk.Frame(self)
        stats.pack(pady=10)
        self.total_students_var = tk.StringVar(value="—")
        self.today_count_var = tk.StringVar(value="—")
        self.ontime_var = tk.StringVar(value="—")
        self.late_var = tk.StringVar(value="—")
        for label, var in (("Active students", self.total_students_var),
                            ("Marked present today", self.today_count_var),
                            ("On time today", self.ontime_var),
                            ("Late today", self.late_var)):
            card = tk.Frame(stats, relief="groove", borderwidth=1, padx=18, pady=10)
            card.pack(side="left", padx=10)
            tk.Label(card, textvariable=var, font=("Segoe UI", 20, "bold")).pack()
            tk.Label(card, text=label, font=("Segoe UI", 9)).pack()

        ttk.Button(self, text="Refresh", command=self._refresh_stats).pack(pady=6)
        late_bar = tk.Frame(self)
        late_bar.pack(pady=6)
        tk.Label(late_bar, text="Late after (HH:MM, 24h):").pack(side="left")
        self.late_time_var = tk.StringVar()
        tk.Entry(late_bar, textvariable=self.late_time_var, width=7).pack(side="left", padx=6)
        ttk.Button(late_bar, text="Set", command=self._save_late_time).pack(side="left")

        tk.Label(self, text="Quick actions", font=("Segoe UI", 12, "bold")).pack(pady=(20, 5))
        actions = tk.Frame(self)
        actions.pack(pady=5)
        buttons = [
            ("Register a student", 1),
            ("Manage students", 2),
            ("View attendance records", 3),
            ("Create an account (Admin/Teacher/Student)", 4),
        ]
        for text, index in buttons:
            ttk.Button(actions, text=text, width=38,
                       command=lambda i=index: self.notebook.select(i)).pack(pady=4)

        self._refresh_stats()

    def _refresh_stats(self):
        students = self.app.conn.call("list_students")
        if students.get("ok"):
            self.total_students_var.set(str(len(students["students"])))
        today = self.app.conn.call("list_attendance_today")
        if today.get("ok"):
            records = today["records"]
            self.today_count_var.set(str(len(records)))
            self.ontime_var.set(str(sum(1 for r in records if r["status"] == "On time")))
            self.late_var.set(str(sum(1 for r in records if r["status"] == "Late")))
        cutoff = self.app.conn.call("get_late_time")
        if cutoff.get("ok"):
            self.late_time_var.set(cutoff["late_time"])

    def _save_late_time(self):
        result = self.app.conn.call("set_late_time", {"late_time": self.late_time_var.get().strip()})
        if result.get("ok"):
            messagebox.showinfo("Saved", "Late cutoff updated. It applies to attendance marked from now on.")
            self._refresh_stats()
        else:
            messagebox.showerror("Failed", result.get("error", "Unknown error"))
    


class RegisterTab(tk.Frame):
    def __init__(self, parent, app: App):
        super().__init__(parent)
        self.app = app
        ttk.Button(self, text="← Back to Dashboard", command=lambda: parent.select(0)).pack(anchor="w", padx=10, pady=(6, 0))

        form = tk.Frame(self)
        form.pack(pady=10)
        self.name_var, self.roll_var, self.dept_var, self.sem_var = (tk.StringVar() for _ in range(4))
        for i, (label, var) in enumerate((("Name", self.name_var), ("Roll no", self.roll_var),
                                           ("Department", self.dept_var), ("Semester", self.sem_var))):
            tk.Label(form, text=label).grid(row=0, column=i * 2, padx=3)
            tk.Entry(form, textvariable=var, width=10).grid(row=0, column=i * 2 + 1, padx=3)

        btns = tk.Frame(self)
        btns.pack(pady=6)
        ttk.Button(btns, text="Register + capture face", command=self._register).pack(side="left", padx=4)
        ttk.Button(btns, text="Train model", command=self._train).pack(side="left", padx=4)

        tk.Label(self, text="Register a student here, then switch to \"Manage Students\" to\n"
                             "search, edit, or deactivate anyone already on the roster.",
                 fg="#555", justify="center").pack(pady=20)

    def _register(self):
        try:
            semester = int(self.sem_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Semester must be a number.")
            return
        if not self.name_var.get() or not self.roll_var.get():
            messagebox.showerror("Invalid input", "Name and roll no are required.")
            return
        result = self.app.conn.call("register_student", {
            "name": self.name_var.get(), "roll_no": self.roll_var.get(),
            "department": self.dept_var.get(), "semester": semester,
        })
        if not result.get("ok"):
            messagebox.showerror("Failed", result.get("error", "Unknown error"))
            return
        self._capture_samples(result["student_id"])

    def _capture_samples(self, student_id):
        # Guided capture: several poses, with live checks for light, distance
        # and blur, so the stored images are varied AND usable.
        cap = cv2.VideoCapture(0)
        total = sum(n for _, n in CAPTURE_STEPS)
        count = 0

        progress_win = tk.Toplevel(self)
        progress_win.title("Capturing face samples")
        progress_var = tk.StringVar(value=f"Captured 0 / {total}")
        tk.Label(progress_win, text="Follow the instructions on the camera window",
                 font=("Segoe UI", 11)).pack(padx=30, pady=(20, 5))
        tk.Label(progress_win, textvariable=progress_var, font=("Segoe UI", 13, "bold")).pack(pady=5)
        bar = ttk.Progressbar(progress_win, length=260, maximum=total)
        bar.pack(padx=30, pady=(5, 20))
        progress_win.update()

        def show(frame, line1, line2=""):
            cv2.putText(frame, line1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if line2:
                cv2.putText(frame, line2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Capturing - press q to stop early", frame)
            return cv2.waitKey(1) & 0xFF == ord("q")

        stopped = False
        for step_no, (instruction, quota) in enumerate(CAPTURE_STEPS, start=1):
            label = f"Step {step_no}/{len(CAPTURE_STEPS)}: {instruction}"

            ready_until = time.time() + 2.0  # time to change pose before capturing
            while time.time() < ready_until and not stopped:
                ok, frame = cap.read()
                if not ok:
                    stopped = True
                    break
                stopped = show(frame, f"GET READY - {label}")

            taken, last_save = 0, 0.0
            while taken < quota and not stopped:
                ok, frame = cap.read()
                if not ok:
                    stopped = True
                    break
                problem, box = check_capture_quality(frame)
                if problem is None and time.time() - last_save >= 0.3:
                    result = self.app.conn.call("add_face_sample", {
                        "student_id": student_id, "image": frame_to_b64(frame),
                    })
                    if result.get("face_found"):
                        taken += 1
                        count += 1
                        last_save = time.time()
                        progress_var.set(f"Captured {count} / {total}")
                        bar["value"] = count
                        progress_win.update_idletasks()
                    else:
                        problem = "Face not clear - face the camera"
                if box is not None:
                    x, y, w, h = box
                    cv2.rectangle(frame, (x, y), (x + w, y + h),
                                  (0, 0, 255) if problem else (0, 255, 0), 2)
                stopped = show(frame, label, problem or "")
            if stopped:
                break

        cap.release()
        cv2.destroyAllWindows()
        progress_win.destroy()
        messagebox.showinfo("Done", f"Captured {count} samples. The model must be trained to include "
                                     "them (Re-capture does this automatically).")
    def _train(self):
        result = self.app.conn.call("train_model")
        if result.get("ok"):
            messagebox.showinfo("Trained", f"Model retrained on {result['image_count']} images "
                                 f"across {result['student_count']} student(s).")
        else:
            messagebox.showerror("Failed", result.get("error", "Unknown error"))


class ManageStudentsTab(tk.Frame):
    """Search the roster, edit a student's details, or deactivate/reactivate
    them. Deactivating never deletes anything - see database.py's
    set_student_active() docstring - it just hides them from the default
    list and stops them being marked present going forward."""

    def __init__(self, parent, app: App):
        super().__init__(parent)
        self.app = app
        ttk.Button(self, text="← Back to Dashboard", command=lambda: parent.select(0)).pack(anchor="w", padx=10, pady=(6, 0))
        self.selected_id = None

        search_bar = tk.Frame(self)
        search_bar.pack(pady=8, fill="x", padx=10)
        tk.Label(search_bar, text="Search (name or roll no):").pack(side="left")
        self.query_var = tk.StringVar()
        entry = tk.Entry(search_bar, textvariable=self.query_var, width=25)
        entry.pack(side="left", padx=6)
        entry.bind("<Return>", lambda e: self._search())
        ttk.Button(search_bar, text="Search", command=self._search).pack(side="left", padx=3)
        ttk.Button(search_bar, text="Show all", command=self._load_all).pack(side="left", padx=3)
        self.show_inactive_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(search_bar, text="Include deactivated", variable=self.show_inactive_var,
                         command=self._load_all).pack(side="left", padx=10)

        self.tree = ttk.Treeview(self, columns=("name", "roll", "dept", "sem", "status"),
                                  show="headings", height=8)
        for col, head in (("name", "Name"), ("roll", "Roll no"), ("dept", "Dept."),
                           ("sem", "Sem."), ("status", "Status")):
            self.tree.heading(col, text=head)
        self.tree.pack(fill="both", expand=True, padx=10, pady=6)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        edit_bar = tk.Frame(self)
        edit_bar.pack(pady=6)
        self.edit_name, self.edit_roll, self.edit_dept, self.edit_sem = (tk.StringVar() for _ in range(4))
        for i, (label, var) in enumerate((("Name", self.edit_name), ("Roll no", self.edit_roll),
                                           ("Department", self.edit_dept), ("Semester", self.edit_sem))):
            tk.Label(edit_bar, text=label).grid(row=0, column=i * 2, padx=3)
            tk.Entry(edit_bar, textvariable=var, width=10).grid(row=0, column=i * 2 + 1, padx=3)

        btns = tk.Frame(self)
        btns.pack(pady=4)
        ttk.Button(btns, text="Save changes", command=self._save_edit).pack(side="left", padx=4)
        ttk.Button(btns, text="Deactivate", command=lambda: self._set_active(False)).pack(side="left", padx=4)
        ttk.Button(btns, text="Reactivate", command=lambda: self._set_active(True)).pack(side="left", padx=4)
        ttk.Button(btns, text="Mark present today", command=self._mark_present).pack(side="left", padx=4)
        ttk.Button(btns, text="Re-capture face", command=self._recapture).pack(side="left", padx=4)

        self._students_by_row = {}
        self._load_all()

    def _populate(self, students):
        for row in self.tree.get_children():
            self.tree.delete(row)
        self._students_by_row.clear()
        for s in students:
            status = "Active" if s["is_active"] else "Deactivated"
            row_id = self.tree.insert("", "end", values=(s["name"], s["roll_no"], s["department"],
                                                           s["semester"], status))
            self._students_by_row[row_id] = s

    def _load_all(self):
        result = self.app.conn.call("list_students", {"include_inactive": self.show_inactive_var.get()})
        if result.get("ok"):
            self._populate(result["students"])

    def _search(self):
        query = self.query_var.get().strip()
        if not query:
            self._load_all()
            return
        result = self.app.conn.call("search_students",
                                     {"query": query, "include_inactive": self.show_inactive_var.get()})
        if result.get("ok"):
            self._populate(result["students"])

    def _on_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        s = self._students_by_row.get(sel[0])
        if not s:
            return
        self.selected_id = s["id"]
        self.edit_name.set(s["name"]); self.edit_roll.set(s["roll_no"])
        self.edit_dept.set(s["department"]); self.edit_sem.set(str(s["semester"]))

    def _save_edit(self):
        if self.selected_id is None:
            messagebox.showinfo("No selection", "Select a student from the table first.")
            return
        try:
            semester = int(self.edit_sem.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Semester must be a number.")
            return
        result = self.app.conn.call("update_student", {
            "student_id": self.selected_id, "name": self.edit_name.get(),
            "roll_no": self.edit_roll.get(), "department": self.edit_dept.get(), "semester": semester,
        })
        if result.get("ok"):
            messagebox.showinfo("Saved", "Student details updated.")
            self._load_all()
        else:
            messagebox.showerror("Failed", result.get("error", "Unknown error"))

    def _set_active(self, active):
        if self.selected_id is None:
            messagebox.showinfo("No selection", "Select a student from the table first.")
            return
        result = self.app.conn.call("set_student_active", {"student_id": self.selected_id, "active": active})
        if result.get("ok"):
            self._load_all()
        else:
            messagebox.showerror("Failed", result.get("error", "Unknown error"))

    def _mark_present(self):
        if self.selected_id is None:
            messagebox.showinfo("No selection", "Select a student from the table first.")
            return
        result = self.app.conn.call("mark_attendance_manual", {"student_id": self.selected_id})
        if not result.get("ok"):
            messagebox.showerror("Failed", result.get("error", "Unknown error"))
            return
        note = "marked present" if result["marked"] else "was already marked present today"
        messagebox.showinfo("Attendance", f"{result['name']} {note}.")

    def _recapture(self):
        if self.selected_id is None:
            messagebox.showinfo("No selection", "Select a student from the table first.")
            return
        replace = messagebox.askyesnocancel(
            "Re-capture face",
            "Delete the OLD face images first?\n\n"
            "Yes = replace old images with new ones (best if they stopped being recognized)\n"
            "No = keep old images and add new ones\n"
            "Cancel = do nothing")
        if replace is None:
            return
        if replace:
            self.app.conn.call("clear_face_samples", {"student_id": self.selected_id})
        RegisterTab._capture_samples(self, self.selected_id)
        result = self.app.conn.call("train_model")
        if result.get("ok"):
            messagebox.showinfo("Retrained", "Model retrained with the new images.")
        else:
            messagebox.showerror("Failed", result.get("error", "Unknown error"))


class AttendanceTab(tk.Frame):
    """Shared by Admin and Teacher. Defaults to full attendance history
    (every record ever taken, most recent first) so nothing is missing by
    default - with a search box to filter by name/roll no, an optional
    specific-date lookup, and CSV export of whatever's currently shown."""

    def __init__(self, parent, app: App):
        super().__init__(parent)
        self.app = app
        if app.user["role"] == "admin":
            ttk.Button(self, text="← Back to Dashboard", command=lambda: parent.select(0)).pack(anchor="w", padx=10, pady=(6, 0))

        search_bar = tk.Frame(self)
        search_bar.pack(pady=6, fill="x", padx=10)
        tk.Label(search_bar, text="Search (name or roll no):").pack(side="left")
        self.query_var = tk.StringVar()
        entry = tk.Entry(search_bar, textvariable=self.query_var, width=20)
        entry.pack(side="left", padx=6)
        entry.bind("<Return>", lambda e: self._search())
        ttk.Button(search_bar, text="Search", command=self._search).pack(side="left", padx=3)

        date_bar = tk.Frame(self)
        date_bar.pack(pady=4, fill="x", padx=10)
        tk.Label(date_bar, text="Or a specific date (YYYY-MM-DD):").pack(side="left")
        self.date_var = tk.StringVar()
        tk.Entry(date_bar, textvariable=self.date_var, width=12).pack(side="left", padx=6)
        ttk.Button(date_bar, text="Load date", command=self._load_date).pack(side="left", padx=3)
        ttk.Button(date_bar, text="Today", command=self._load_today).pack(side="left", padx=3)
        ttk.Button(date_bar, text="Show all history", command=self._load_all).pack(side="left", padx=10)
        ttk.Button(date_bar, text="Export to CSV", command=self._export_csv).pack(side="left", padx=3)

        self.tree = ttk.Treeview(self, columns=("date", "name", "roll", "time", "status"), show="headings", height=12)
        for col, head in (("date", "Date"), ("name", "Name"), ("roll", "Roll no"), ("time", "Time"), ("status", "Status")):
            self.tree.heading(col, text=head)
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)
        self.tree.tag_configure("late", foreground="#c00000")

        self._current_records = []
        self._load_all()

    def _populate(self, records):
        self._current_records = records
        for row in self.tree.get_children():
            self.tree.delete(row)
        for rec in records:
            # "date" is present for full-history/search results but not for
            # the today/specific-date endpoints (which only return time,
            # since the date is implied) - fall back to what was searched.
            day = rec.get("date", self.date_var.get() or "")
            self.tree.insert("", "end", values=(day, rec["name"], rec["roll_no"], rec["time"], rec.get("status", "")),
                             tags=("late",) if rec.get("status") == "Late" else ())
    def _load_all(self):
        self.date_var.set(""); self.query_var.set("")
        result = self.app.conn.call("list_attendance_all")
        if result.get("ok"):
            self._populate(result["records"])

    def _load_today(self):
        self.date_var.set(""); self.query_var.set("")
        result = self.app.conn.call("list_attendance_today")
        if result.get("ok"):
            import datetime
            today = datetime.date.today().isoformat()
            self._populate([{**r, "date": today} for r in result["records"]])

    def _load_date(self):
        day = self.date_var.get().strip()
        if not day:
            self._load_today()
            return
        result = self.app.conn.call("list_attendance_for_date", {"date": day})
        if not result.get("ok"):
            messagebox.showerror("Failed", result.get("error", "Unknown error"))
            return
        self._populate([{**r, "date": day} for r in result["records"]])
        if not result["records"]:
            messagebox.showinfo("No records", f"No attendance found for {day}.")

    def _search(self):
        query = self.query_var.get().strip()
        if not query:
            self._load_all()
            return
        result = self.app.conn.call("search_attendance", {"query": query})
        if result.get("ok"):
            self._populate(result["records"])

    def _export_csv(self):
        if not self._current_records:
            messagebox.showinfo("Nothing to export", "There are no records currently shown to export.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV files", "*.csv")],
                                             initialfile="attendance_export.csv")
        if not path:
            return
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Name", "Roll No", "Time", "Status"])
            for row_id in self.tree.get_children():
                writer.writerow(self.tree.item(row_id)["values"])
        messagebox.showinfo("Exported", f"Saved to {path}")


class ManageAccountsTab(tk.Frame):
    """Create login accounts (Admin, Teacher, or Student) and see what
    accounts already exist. This is the piece that was completely missing
    before - create_user existed on the server with no way to reach it."""

    def __init__(self, parent, app: App):
        super().__init__(parent)
        self.app = app
        ttk.Button(self, text="← Back to Dashboard", command=lambda: parent.select(0)).pack(anchor="w", padx=10, pady=(6, 0))

        form = tk.Frame(self)
        form.pack(pady=10)
        self.username_var, self.password_var, self.roll_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
        self.role_var = tk.StringVar(value="student")

        tk.Label(form, text="Username").grid(row=0, column=0, padx=4, pady=4, sticky="e")
        tk.Entry(form, textvariable=self.username_var).grid(row=0, column=1, padx=4, pady=4)
        tk.Label(form, text="Password").grid(row=1, column=0, padx=4, pady=4, sticky="e")
        tk.Entry(form, textvariable=self.password_var, show="*").grid(row=1, column=1, padx=4, pady=4)
        tk.Label(form, text="Role").grid(row=2, column=0, padx=4, pady=4, sticky="e")
        role_menu = tk.Frame(form)
        role_menu.grid(row=2, column=1, sticky="w")
        for role in ("admin", "teacher", "student"):
            ttk.Radiobutton(role_menu, text=role.capitalize(), variable=self.role_var, value=role,
                             command=self._toggle_roll_field).pack(side="left")
        self.roll_label = tk.Label(form, text="Link to student (roll no)")
        self.roll_label.grid(row=3, column=0, padx=4, pady=4, sticky="e")
        self.roll_entry = tk.Entry(form, textvariable=self.roll_var)
        self.roll_entry.grid(row=3, column=1, padx=4, pady=4)

        ttk.Button(self, text="Create account", command=self._create).pack(pady=8)

        self.tree = ttk.Treeview(self, columns=("username", "role", "student"),
                                  show="headings", height=8)
        for col, head in (("username", "Username"), ("role", "Role"), ("student", "Linked student")):
            self.tree.heading(col, text=head)
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)
        ttk.Button(self, text="Refresh list", command=self._refresh).pack(pady=4)

        self._toggle_roll_field()
        self._refresh()

    def _toggle_roll_field(self):
        state = "normal" if self.role_var.get() == "student" else "disabled"
        self.roll_entry.configure(state=state)

    def _create(self):
        username, password = self.username_var.get().strip(), self.password_var.get()
        if not username or not password:
            messagebox.showerror("Invalid input", "Username and password are required.")
            return
        payload = {"username": username, "password": password, "role": self.role_var.get()}

        if self.role_var.get() == "student":
            roll = self.roll_var.get().strip()
            if not roll:
                messagebox.showerror("Invalid input", "Enter the roll no to link this login to.")
                return
            everyone = self.app.conn.call("list_students", {"include_inactive": True})
            canon = lambda r: (r.strip().lstrip("0") or "0") if r.strip().isdigit() else r.strip().upper()
            matches = [s for s in everyone.get("students", []) if canon(s["roll_no"]) == canon(roll)]
            if not matches:
                messagebox.showerror("Not found", f"No student with roll no '{roll}'.")
                return
            payload["student_id"] = matches[0]["id"]

        result = self.app.conn.call("create_user", payload)
        if result.get("ok"):
            messagebox.showinfo("Created", f"Account '{username}' created.")
            self.username_var.set(""); self.password_var.set(""); self.roll_var.set("")
            self._refresh()
        else:
            messagebox.showerror("Failed", result.get("error", "Unknown error"))

    def _refresh(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        result = self.app.conn.call("list_users")
        if not result.get("ok"):
            return
        for u in result["users"]:
            linked = f"{u['student_name']} ({u['student_roll']})" if u["student_name"] else "-"
            self.tree.insert("", "end", values=(u["username"], u["role"].capitalize(), linked))


class TeacherFrame(tk.Frame):
    def __init__(self, parent, app: App):
        super().__init__(parent)
        self.app = app
        tk.Label(self, text=f"Teacher — {app.user['username']}", font=("Segoe UI", 15)).pack(pady=8)
        ttk.Button(self, text="← Logout", command=app.logout).pack(anchor="w", padx=10)

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)
        notebook.add(AttendanceTab(notebook, app), text="Attendance Records")
        notebook.add(TeacherStudentLookupTab(notebook, app), text="Student Lookup")


class TeacherStudentLookupTab(tk.Frame):
    """For a Teacher: search the roster and mark a student present
    manually (useful when recognition doesn't work for someone). No
    edit/deactivate buttons here, though - those stay Admin-only, and
    that boundary is enforced server-side regardless of what buttons
    this screen does or doesn't show."""

    def __init__(self, parent, app: App):
        super().__init__(parent)
        self.app = app
        self.selected_id = None

        search_bar = tk.Frame(self)
        search_bar.pack(pady=8, fill="x", padx=10)
        tk.Label(search_bar, text="Search (name or roll no):").pack(side="left")
        self.query_var = tk.StringVar()
        entry = tk.Entry(search_bar, textvariable=self.query_var, width=25)
        entry.pack(side="left", padx=6)
        entry.bind("<Return>", lambda e: self._search())
        ttk.Button(search_bar, text="Search", command=self._search).pack(side="left", padx=3)
        ttk.Button(search_bar, text="Show all", command=self._load_all).pack(side="left", padx=3)

        self.tree = ttk.Treeview(self, columns=("name", "roll", "dept", "sem"), show="headings", height=10)
        for col, head in (("name", "Name"), ("roll", "Roll no"), ("dept", "Dept."), ("sem", "Sem.")):
            self.tree.heading(col, text=head)
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        ttk.Button(self, text="Mark present today", command=self._mark_present).pack(pady=6)

        self._students_by_row = {}
        self._load_all()

    def _populate(self, students):
        for row in self.tree.get_children():
            self.tree.delete(row)
        self._students_by_row.clear()
        for s in students:
            row_id = self.tree.insert("", "end", values=(s["name"], s["roll_no"], s["department"], s["semester"]))
            self._students_by_row[row_id] = s

    def _on_select(self, event):
        sel = self.tree.selection()
        if sel:
            self.selected_id = self._students_by_row[sel[0]]["id"]

    def _mark_present(self):
        if self.selected_id is None:
            messagebox.showinfo("No selection", "Select a student from the table first.")
            return
        result = self.app.conn.call("mark_attendance_manual", {"student_id": self.selected_id})
        if not result.get("ok"):
            messagebox.showerror("Failed", result.get("error", "Unknown error"))
            return
        note = "marked present" if result["marked"] else "was already marked present today"
        messagebox.showinfo("Attendance", f"{result['name']} {note}.")

    def _load_all(self):
        result = self.app.conn.call("list_students")
        if result.get("ok"):
            self._populate(result["students"])

    def _search(self):
        query = self.query_var.get().strip()
        if not query:
            self._load_all()
            return
        result = self.app.conn.call("search_students", {"query": query})
        if result.get("ok"):
            self._populate(result["students"])


class StudentFrame(tk.Frame):
    def __init__(self, parent, app: App):
        super().__init__(parent)
        self.app = app
        tk.Label(self, text=f"My Attendance — {app.user['username']}", font=("Segoe UI", 15)).pack(pady=8)
        ttk.Button(self, text="← Logout", command=app.logout).pack(anchor="w", padx=10)
        self.tree = ttk.Treeview(self, columns=("date", "time", "status"), show="headings", height=12)
        for col, head in (("date", "Date"), ("time", "Time"), ("status", "Status")):
            self.tree.heading(col, text=head)
        self.tree.pack(fill="both", expand=True, padx=10, pady=10)
        self._refresh()

    def _refresh(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        result = self.app.conn.call("my_attendance")
        if not result.get("ok"):
            messagebox.showerror("Failed", result.get("error", "Unknown error"))
            return
        for rec in result.get("records", []):
            self.tree.insert("", "end", values=(rec["date"], rec["time"], rec["status"]))


if __name__ == "__main__":
    app = App()
    app.mainloop()
