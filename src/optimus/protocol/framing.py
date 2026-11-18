"""
TCP Framing and message serialization for Optimus protocol.
Uses a 4-byte big-endian length prefix followed by serialized Protocol Buffer bytes.
"""

import socket
import struct
import time
import uuid
from typing import Optional

from optimus.protocol.generated.messages_pb2 import Envelope

DEFAULT_MAX_MESSAGE_SIZE = 4 * 1024 * 1024  # 4 MiB
HEADER_FORMAT = ">I"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
CURRENT_PROTOCOL_VERSION = 1


class ProtocolError(Exception):
    """Raised when protocol framing or validation fails."""
    pass


class MessageTooLargeError(ProtocolError):
    """Raised when message exceeds max message size limit."""
    pass


def create_envelope(
    payload_field_name: str,
    payload_obj,
    sender_id: str,
    protocol_version: int = CURRENT_PROTOCOL_VERSION,
    auth_token: str = "",
) -> Envelope:
    """Helper to construct an Envelope with metadata and a single payload."""
    env = Envelope()
    env.protocol_version = protocol_version
    env.message_id = str(uuid.uuid4())
    env.sender_id = sender_id
    env.timestamp_ms = int(time.time() * 1000)
    env.auth_token = auth_token
    getattr(env, payload_field_name).CopyFrom(payload_obj)
    return env


def send_message(
    sock: socket.socket,
    envelope: Envelope,
    max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
) -> None:
    """
    Serializes and sends an Envelope message preceded by a 4-byte big-endian length prefix.
    Ensures all bytes are sent completely even with partial socket writes.
    """
    payload_bytes = envelope.SerializeToString()
    length = len(payload_bytes)

    if length > max_message_size:
        raise MessageTooLargeError(
            f"Message size {length} bytes exceeds maximum allowed {max_message_size} bytes"
        )

    header = struct.pack(HEADER_FORMAT, length)
    full_data = header + payload_bytes
    sock.sendall(full_data)


def recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    """
    Reads exactly num_bytes from a socket.
    Raises ConnectionError if connection closes prematurely.
    """
    chunks = []
    received = 0
    while received < num_bytes:
        chunk = sock.recv(num_bytes - received)
        if not chunk:
            if received == 0:
                # Clean EOF at message boundary
                raise ConnectionResetError("Connection closed by peer")
            raise ConnectionError(
                f"Unexpected EOF: expected {num_bytes} bytes, received {received} bytes"
            )
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def recv_message(
    sock: socket.socket,
    max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
    expected_protocol_version: Optional[int] = CURRENT_PROTOCOL_VERSION,
) -> Envelope:
    """
    Reads a length-prefixed Protocol Buffers Envelope from the socket.
    Validates message size and protocol version.
    """
    header_bytes = recv_exact(sock, HEADER_SIZE)
    (length,) = struct.unpack(HEADER_FORMAT, header_bytes)

    if length > max_message_size:
        raise MessageTooLargeError(
            f"Incoming payload length {length} exceeds maximum allowed size {max_message_size}"
        )

    payload_bytes = recv_exact(sock, length)
    envelope = Envelope()
    envelope.ParseFromString(payload_bytes)

    if (
        expected_protocol_version is not None
        and envelope.protocol_version != expected_protocol_version
    ):
        raise ProtocolError(
            f"Protocol version mismatch: expected {expected_protocol_version}, got {envelope.protocol_version}"
        )

    return envelope
