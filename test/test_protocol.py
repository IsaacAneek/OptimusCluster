"""
Tests for Optimus TCP Framing, Message Serialization, and Protocol Buffers Envelope.
"""

import socket
import struct
import threading
import time
import pytest

from optimus.protocol.framing import (
    CURRENT_PROTOCOL_VERSION,
    DEFAULT_MAX_MESSAGE_SIZE,
    MessageTooLargeError,
    ProtocolError,
    create_envelope,
    recv_exact,
    recv_message,
    send_message,
)
from optimus.protocol.generated.messages_pb2 import (
    Envelope,
    StatusRequest,
    StatusResponse,
    WorkerState,
)


def test_create_envelope():
    req = StatusRequest(probe_only=True)
    env = create_envelope(
        payload_field_name="status_request",
        payload_obj=req,
        sender_id="controller-01",
        protocol_version=1,
        auth_token="secret-123",
    )
    assert env.protocol_version == 1
    assert env.sender_id == "controller-01"
    assert env.auth_token == "secret-123"
    assert env.WhichOneof("payload") == "status_request"
    assert env.status_request.probe_only is True


def test_framing_send_recv_roundtrip():
    server_sock, client_sock = socket.socketpair()
    try:
        req = StatusRequest(probe_only=True)
        env = create_envelope("status_request", req, sender_id="test-client")

        # Send from client to server
        send_message(client_sock, env)

        # Receive on server
        received_env = recv_message(server_sock)
        assert received_env.sender_id == "test-client"
        assert received_env.WhichOneof("payload") == "status_request"
        assert received_env.status_request.probe_only is True
    finally:
        server_sock.close()
        client_sock.close()


def test_framing_multiple_messages_on_single_connection():
    server_sock, client_sock = socket.socketpair()
    try:
        messages = []
        for i in range(5):
            resp = StatusResponse(
                worker_id=f"worker-{i}",
                hostname=f"host-{i}.local",
                protocol_version=CURRENT_PROTOCOL_VERSION,
                status=WorkerState.AVAILABLE,
                active_job_count=i,
            )
            env = create_envelope("status_response", resp, sender_id=f"worker-{i}")
            messages.append(env)
            send_message(client_sock, env)

        for i in range(5):
            rec_env = recv_message(server_sock)
            assert rec_env.sender_id == f"worker-{i}"
            assert rec_env.status_response.worker_id == f"worker-{i}"
            assert rec_env.status_response.active_job_count == i
    finally:
        server_sock.close()
        client_sock.close()


def test_framing_partial_reads():
    """Simulates network fragmentation where bytes arrive in small slices."""
    server_sock, client_sock = socket.socketpair()
    try:
        req = StatusRequest(probe_only=False)
        env = create_envelope("status_request", req, sender_id="fragment-sender")
        payload = env.SerializeToString()
        header = struct.pack(">I", len(payload))
        full_bytes = header + payload

        def send_fragmented():
            for byte_idx in range(len(full_bytes)):
                client_sock.sendall(full_bytes[byte_idx : byte_idx + 1])
                time.sleep(0.001)

        t = threading.Thread(target=send_fragmented)
        t.start()

        received_env = recv_message(server_sock)
        t.join()

        assert received_env.sender_id == "fragment-sender"
        assert received_env.status_request.probe_only is False
    finally:
        server_sock.close()
        client_sock.close()


def test_framing_message_too_large():
    server_sock, client_sock = socket.socketpair()
    try:
        req = StatusRequest(probe_only=True)
        env = create_envelope("status_request", req, sender_id="oversized")
        # Set max_message_size to smaller than envelope
        with pytest.raises(MessageTooLargeError):
            send_message(client_sock, env, max_message_size=5)
    finally:
        server_sock.close()
        client_sock.close()


def test_framing_version_mismatch():
    server_sock, client_sock = socket.socketpair()
    try:
        req = StatusRequest(probe_only=True)
        env = create_envelope("status_request", req, sender_id="old-client", protocol_version=99)
        send_message(client_sock, env)

        with pytest.raises(ProtocolError, match="Protocol version mismatch"):
            recv_message(server_sock, expected_protocol_version=1)
    finally:
        server_sock.close()
        client_sock.close()
