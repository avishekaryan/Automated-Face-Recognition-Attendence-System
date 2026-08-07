import socket

HOST = "localhost"
PORT = 17000

roll_no = input("Enter Roll Number : ")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
    try:
        client_socket.connect((HOST, PORT))
        client_socket.sendall(roll_no.encode())
        response = client_socket.recv(4096)
        print("\nStudent Information")
        print("----------------------------")
        print(response.decode())
    except Exception as e:
        print("Error :", e)