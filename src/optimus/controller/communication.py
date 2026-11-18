"""
Optimus Controller Communication Client.
Manages TCP connections to worker nodes and serialized Protocol Buffers exchanges.
"""

import logging
import socket
import threading
import time
from typing import Dict, Optional, Tuple

from optimus.protocol.framing import (
    CURRENT_PROTOCOL_VERSION,
    DEFAULT_MAX_MESSAGE_SIZE,
    create_envelope,
    recv_message,
    send_message,
    ProtocolError,
)
from optimus.protocol.generated.messages_pb2 import Envelope

logger = logging.getLogger("optimus.controller.communication")


class WorkerCommunicationError(Exception):
    """Base error for communication failures with workers."""
    pass


class WorkerConnectionTimeout(WorkerCommunicationError):
    """Raised when connecting or receiving from worker times out."""
    pass


class WorkerClient:
    """Manages a single worker's TCP connection."""

    def __init__(
        self,
        hostname: str,
        port: int,
        timeout: float = 3.0,
        cluster_secret: str = "",
        controller_id: str = "controller",
        max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
    ):
        self.hostname = hostname
        self.port = port
        self.timeout = timeout
        self.cluster_secret = cluster_secret
        self.controller_id = controller_id
        self.max_message_size = max_message_size

        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def _connect(self) -> None:
        if self._sock is not None:
            return

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            sock.connect((self.hostname, self.port))
            self._sock = sock
            logger.debug(f"Connected to worker at {self.hostname}:{self.port}")
        except socket.timeout as e:
            sock.close()
            raise WorkerConnectionTimeout(f"Connection timeout to {self.hostname}:{self.port}: {e}")
        except OSError as e:
            sock.close()
            raise WorkerCommunicationError(f"Failed to connect to {self.hostname}:{self.port}: {e}")

    def send_request(
        self,
        payload_name: str,
        payload_obj,
        timeout: Optional[float] = None,
    ) -> Envelope:
        """
        Sends a typed request payload wrapped in an Envelope and returns the response Envelope.
        Re-establishes connection automatically if previous connection was severed.
        """
        effective_timeout = timeout if timeout is not None else self.timeout
        with self._lock:
            try:
                self._connect()
                self._sock.settimeout(effective_timeout)

                req_env = create_envelope(
                    payload_field_name=payload_name,
                    payload_obj=payload_obj,
                    sender_id=self.controller_id,
                    protocol_version=CURRENT_PROTOCOL_VERSION,
                    auth_token=self.cluster_secret,
                )

                send_message(self._sock, req_env, self.max_message_size)
                resp_env = recv_message(
                    self._sock,
                    max_message_size=self.max_message_size,
                    expected_protocol_version=CURRENT_PROTOCOL_VERSION,
                )
                return resp_env

            except socket.timeout as e:
                self.close()
                raise WorkerConnectionTimeout(f"Timeout waiting for response from {self.hostname}:{self.port}")
            except (ConnectionError, OSError, ProtocolError) as e:
                self.close()
                raise WorkerCommunicationError(f"Communication error with {self.hostname}:{self.port}: {e}")

    def close(self) -> None:
        """Closes the active connection."""
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None


class WorkerClientManager:
    """Pool and manager of WorkerClients keyed by (hostname, port)."""

    def __init__(
        self,
        default_timeout: float = 3.0,
        cluster_secret: str = "",
        controller_id: str = "controller",
        max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
    ):
        self.default_timeout = default_timeout
        self.cluster_secret = cluster_secret
        self.controller_id = controller_id
        self.max_message_size = max_message_size
        self._clients: Dict[Tuple[str, int], WorkerClient] = {}
        self._lock = threading.Lock()

    def get_client(self, hostname: str, port: int) -> WorkerClient:
        key = (hostname, port)
        with self._lock:
            if key not in self._clients:
                self._clients[key] = WorkerClient(
                    hostname=hostname,
                    port=port,
                    timeout=self.default_timeout,
                    cluster_secret=self.cluster_secret,
                    controller_id=self.controller_id,
                    max_message_size=self.max_message_size,
                )
            return self._clients[key]

    def close_all(self) -> None:
        with self._lock:
            for client in self._clients.values():
                client.close()
            self._clients.clear()
