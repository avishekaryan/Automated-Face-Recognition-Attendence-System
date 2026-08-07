import sqlite3

with sqlite3.connect("attendance.db") as conn:
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students(
        roll_no TEXT PRIMARY KEY,
        name TEXT,
        course TEXT,
        semester INTEGER,
        attendance TEXT
    )
    """)

    students = [
        ("AI001", "Abhishek Kumar Sah", "B.Tech AI", 2, "Present"),
        ("AI002", "Bivek Kumar Yadav", "B.Tech AI", 2, "Absent"),
        ("AI003", "Bigyan Padhrya", "B.Tech AI", 2, "Present"),
        ("AI004", "Dipendra Kumar Sah", "B.Tech AI", 2, "Present"),
        ("AI005", "Kushi Chaudhary", "B.Tech AI", 2, "Absent")
    ]

    cursor.executemany(
        "INSERT OR REPLACE INTO students VALUES (?, ?, ?, ?, ?)",
        students
    )
print("Database created successfully.")