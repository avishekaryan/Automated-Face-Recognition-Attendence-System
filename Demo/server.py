import socket
import sqlite3

HOST = "localhost"
PORT = 17000


def search_student(roll_no):

    with sqlite3.connect("attendance.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM students WHERE roll_no=?",
            (roll_no,)
        )
        student = cursor.fetchone()

    if student:
        return (
            f"Roll Number : {student[0]}\n"
            f"Name        : {student[1]}\n"
            f"Course      : {student[2]}\n"
            f"Semester    : {student[3]}\n"
            f"Attendance  : {student[4]}"
        )

    return "Student not found."


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(5)

    print(f"Server running on {HOST}:{PORT}")

    while True:
        client_socket, address = server_socket.accept()
        with client_socket:
            print(f"Connected by {address}")

            try:
                data = client_socket.recv(1024)
                if not data:
                    continue
                roll_no = data.decode().strip()
                result = search_student(roll_no)
                client_socket.sendall(result.encode())
            except Exception as e:
                client_socket.sendall(
                    f"Error : {e}".encode()
                )