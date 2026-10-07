"""
protocol.py
-----------
A TCP socket only gives you a stream of bytes - it doesn't know where one
message ends and the next begins. This wraps every message with a 4-byte
length header so the receiver always knows exactly how many bytes to read
for one complete message. Both the server and every client import these
same two functions, so they always agree on the format.

Message body is JSON, encoded as UTF-8. Binary data (a camera frame) is
base64-encoded inside the JSON so the whole message stays plain text.
"""

import json
import socket
import struct

HEADER_SIZE = 4  # bytes, big-endian unsigned int = message length


def send_msg(sock: socket.socket, obj: dict) -> None:
    body = json.dumps(obj).encode("utf-8")
    header = struct.pack(">I", len(body))
    sock.sendall(header + body)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed while reading")
        buf += chunk
    return buf


def recv_msg(sock: socket.socket) -> dict:
    header = _recv_exact(sock, HEADER_SIZE)
    (length,) = struct.unpack(">I", header)
    body = _recv_exact(sock, length)
    return json.loads(body.decode("utf-8"))
