"""
Optimus Controller Communication Client.
Talks to optimusd worker daemons over the REST/JSON control-plane API
(see ``optimus.protocol.rest``) using only the standard library ``http.client``.
"""

import http.client
import logging
import socket
import threading
from typing import Any, Dict, Iterable, Optional, Tuple, Type, TypeVar
from urllib.parse import urlencode

from optimus.protocol import rest
from optimus.protocol.messages import (
    BaseMessage,
    CacheInventoryResponse,
    CacheSummaryResponse,
    CheckpointListResponse,
    ErrorResponse,
    JobResponse,
    JobStatusResponse,
    JobStop,
    JobSubmit,
    ResourceResponse,
    ShutdownRequest,
    StatusResponse,
)
from optimus.protocol.rest import CURRENT_PROTOCOL_VERSION, DEFAULT_MAX_MESSAGE_SIZE

logger = logging.getLogger("optimus.controller.communication")

M = TypeVar("M", bound=BaseMessage)


class WorkerCommunicationError(Exception):
    """Base error for communication failures with workers."""
    pass


class WorkerConnectionTimeout(WorkerCommunicationError):
    """Raised when connecting or receiving from worker times out."""
    pass


class WorkerRequestError(WorkerCommunicationError):
    """Raised when a worker answers with an HTTP error status (ErrorResponse body)."""

    def __init__(self, status: int, error_code: str, error_message: str):
        super().__init__(f"Worker returned HTTP {status} {error_code}: {error_message}")
        self.status = status
        self.error_code = error_code
        self.error_message = error_message


class WorkerClient:
    """REST client for a single optimusd worker. Thread-safe: each call opens its own connection."""

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

    @property
    def base_url(self) -> str:
        return f"http://{self.hostname}:{self.port}"

    # ----------------------------------------------------------------- core

    def request(
        self,
        method: str,
        path: str,
        query: Optional[Dict[str, Any]] = None,
        body: Any = None,
        timeout: Optional[float] = None,
        ok_statuses: Iterable[int] = (200, 201, 202),
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Performs one HTTP request and returns (status, parsed JSON body).
        Raises WorkerRequestError for non-OK statuses, WorkerConnectionTimeout on
        timeouts, and WorkerCommunicationError on any other transport failure.
        """
        effective_timeout = timeout if timeout is not None else self.timeout
        if query:
            path = f"{path}?{urlencode({k: v for k, v in query.items() if v is not None})}"

        payload = rest.encode(body) if body is not None else None
        if payload is not None and len(payload) > self.max_message_size:
            raise rest.MessageTooLargeError(
                f"Request body of {len(payload)} bytes exceeds maximum {self.max_message_size} bytes"
            )

        headers = {
            "Accept": rest.JSON_CONTENT_TYPE,
            "Connection": "close",
            rest.PROTOCOL_VERSION_HEADER: str(CURRENT_PROTOCOL_VERSION),
            rest.SENDER_HEADER: self.controller_id,
        }
        if payload is not None:
            headers["Content-Type"] = rest.JSON_CONTENT_TYPE
        if self.cluster_secret:
            headers[rest.AUTH_HEADER] = f"{rest.AUTH_SCHEME} {self.cluster_secret}"

        conn = http.client.HTTPConnection(self.hostname, self.port, timeout=effective_timeout)
        try:
            conn.request(method, path, body=payload, headers=headers)
            resp = conn.getresponse()
            length = resp.getheader("Content-Length")
            if length is not None and int(length) > self.max_message_size:
                raise rest.MessageTooLargeError(
                    f"Response of {length} bytes exceeds maximum {self.max_message_size} bytes"
                )
            raw = resp.read(self.max_message_size + 1)
            if len(raw) > self.max_message_size:
                raise rest.MessageTooLargeError(
                    f"Response exceeds maximum {self.max_message_size} bytes"
                )
            status = resp.status
        except socket.timeout as e:
            raise WorkerConnectionTimeout(
                f"Timeout talking to {self.base_url}{path}: {e}"
            )
        except rest.ProtocolError as e:
            raise WorkerCommunicationError(f"Protocol error with {self.base_url}: {e}")
        except (OSError, http.client.HTTPException) as e:
            raise WorkerCommunicationError(f"Communication error with {self.base_url}: {e}")
        finally:
            conn.close()

        try:
            data = rest.decode_json(raw)
        except rest.ProtocolError as e:
            raise WorkerCommunicationError(
                f"Invalid response from {self.base_url}{path} (HTTP {status}): {e}"
            )

        if status not in ok_statuses:
            err = ErrorResponse.from_dict(data)
            raise WorkerRequestError(
                status, err.error_code or "HTTP_ERROR", err.error_message or f"HTTP {status}"
            )
        return status, data

    def _get(self, path: str, cls: Type[M], query=None, timeout=None) -> M:
        _, data = self.request("GET", path, query=query, timeout=timeout)
        return cls.from_dict(data)

    def _post(self, path: str, body, cls: Type[M], timeout=None, ok_statuses=(200, 201, 202)) -> M:
        _, data = self.request("POST", path, body=body, timeout=timeout, ok_statuses=ok_statuses)
        return cls.from_dict(data)

    # ------------------------------------------------------------ endpoints

    def get_status(self, timeout: Optional[float] = None) -> StatusResponse:
        """GET /v1/status"""
        return self._get(rest.STATUS_PATH, StatusResponse, timeout=timeout)

    def get_resources(self, timeout: Optional[float] = None) -> ResourceResponse:
        """GET /v1/resources"""
        return self._get(rest.RESOURCES_PATH, ResourceResponse, timeout=timeout)

    def get_cache_summary(
        self, dataset_id: str, dataset_version: str, timeout: Optional[float] = None
    ) -> CacheSummaryResponse:
        """GET /v1/cache/summary"""
        return self._get(
            rest.CACHE_SUMMARY_PATH,
            CacheSummaryResponse,
            query={"dataset_id": dataset_id, "dataset_version": dataset_version},
            timeout=timeout,
        )

    def get_cache_inventory(
        self,
        dataset_id: str,
        dataset_version: str,
        known_revision: int = 0,
        page: int = 1,
        page_size: int = 500,
        timeout: Optional[float] = None,
    ) -> CacheInventoryResponse:
        """GET /v1/cache/inventory"""
        return self._get(
            rest.CACHE_INVENTORY_PATH,
            CacheInventoryResponse,
            query={
                "dataset_id": dataset_id,
                "dataset_version": dataset_version,
                "known_revision": known_revision,
                "page": page,
                "page_size": page_size,
            },
            timeout=timeout,
        )

    def list_checkpoints(
        self, checkpoint_dir: str = "", job_id: str = "", timeout: Optional[float] = None
    ) -> CheckpointListResponse:
        """GET /v1/checkpoints"""
        query = {}
        if checkpoint_dir:
            query["checkpoint_dir"] = checkpoint_dir
        if job_id:
            query["job_id"] = job_id
        return self._get(rest.CHECKPOINTS_PATH, CheckpointListResponse, query=query, timeout=timeout)

    def submit_job(self, job: JobSubmit, timeout: Optional[float] = None) -> JobResponse:
        """POST /v1/jobs -> 201 accepted, 409 rejected (both return a JobResponse)."""
        return self._post(
            rest.JOBS_PATH, job, JobResponse, timeout=timeout, ok_statuses=(200, 201, 409)
        )

    def get_job_status(self, job_id: str, timeout: Optional[float] = None) -> JobStatusResponse:
        """GET /v1/jobs/{job_id}"""
        return self._get(rest.job_path(job_id), JobStatusResponse, timeout=timeout)

    def stop_job(
        self, job_id: str, reason: str = "", graceful: bool = True, timeout: Optional[float] = None
    ) -> StatusResponse:
        """POST /v1/jobs/{job_id}/stop"""
        body = JobStop(job_id=job_id, reason=reason, graceful=graceful)
        return self._post(rest.job_stop_path(job_id), body, StatusResponse, timeout=timeout)

    def shutdown(
        self, reason: str = "", force: bool = False, timeout: Optional[float] = None
    ) -> StatusResponse:
        """POST /v1/shutdown"""
        body = ShutdownRequest(reason=reason, force=force)
        return self._post(rest.SHUTDOWN_PATH, body, StatusResponse, timeout=timeout)

    def close(self) -> None:
        """No persistent connection is held; kept for API compatibility."""
        return None


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
