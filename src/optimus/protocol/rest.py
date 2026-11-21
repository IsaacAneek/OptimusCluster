"""
Optimus control-plane REST API definitions.

The controller talks to each ``optimusd`` worker daemon over plain HTTP/1.1 with
JSON bodies. This module holds the pieces shared by the server
(``optimus.worker.server``) and the client (``optimus.controller.communication``):
route paths, header names, and JSON encode/decode helpers.

Endpoints (all under ``/v1``):

    GET  /v1/status                      -> StatusResponse
    GET  /v1/resources                   -> ResourceResponse
    GET  /v1/cache/summary               -> CacheSummaryResponse
         ?dataset_id=&dataset_version=
    GET  /v1/cache/inventory             -> CacheInventoryResponse
         ?dataset_id=&dataset_version=&known_revision=&page=&page_size=
    GET  /v1/checkpoints                 -> CheckpointListResponse
         ?checkpoint_dir=&job_id=
    POST /v1/jobs            (JobSubmit) -> 201 JobResponse | 409 JobResponse (rejected)
    GET  /v1/jobs/{job_id}               -> JobStatusResponse
    POST /v1/jobs/{job_id}/stop (JobStop)-> StatusResponse
    POST /v1/shutdown  (ShutdownRequest) -> 202 StatusResponse

Errors are returned as an ``ErrorResponse`` JSON body with a 4xx/5xx status.
When a cluster secret is configured, every request must carry
``Authorization: Bearer <secret>``.

Gloo / torch.distributed training traffic does NOT use this API.
"""

import json
from typing import Any, Dict, Type, TypeVar

from optimus.protocol.messages import BaseMessage

API_PREFIX = "/v1"
CURRENT_PROTOCOL_VERSION = 1
DEFAULT_MAX_MESSAGE_SIZE = 4 * 1024 * 1024  # 4 MiB, max request/response body

PROTOCOL_VERSION_HEADER = "X-Optimus-Protocol-Version"
SENDER_HEADER = "X-Optimus-Sender"
AUTH_HEADER = "Authorization"
AUTH_SCHEME = "Bearer"
JSON_CONTENT_TYPE = "application/json"

# Route paths
STATUS_PATH = f"{API_PREFIX}/status"
RESOURCES_PATH = f"{API_PREFIX}/resources"
CACHE_SUMMARY_PATH = f"{API_PREFIX}/cache/summary"
CACHE_INVENTORY_PATH = f"{API_PREFIX}/cache/inventory"
CHECKPOINTS_PATH = f"{API_PREFIX}/checkpoints"
JOBS_PATH = f"{API_PREFIX}/jobs"
SHUTDOWN_PATH = f"{API_PREFIX}/shutdown"


def job_path(job_id: str) -> str:
    from urllib.parse import quote

    return f"{JOBS_PATH}/{quote(job_id, safe='')}"


def job_stop_path(job_id: str) -> str:
    return f"{job_path(job_id)}/stop"


class ProtocolError(Exception):
    """Raised when a request/response cannot be parsed or violates the API contract."""


class MessageTooLargeError(ProtocolError):
    """Raised when a body exceeds the configured maximum size."""


M = TypeVar("M", bound=BaseMessage)


def encode(message: Any) -> bytes:
    """Serialize a message dataclass (or plain dict) to UTF-8 JSON bytes."""
    data = message.to_dict() if hasattr(message, "to_dict") else message
    return json.dumps(data).encode("utf-8")


def decode_json(raw: bytes) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"Invalid JSON body: {e}")
    if not isinstance(data, dict):
        raise ProtocolError("JSON body must be an object")
    return data


def decode(raw: bytes, cls: Type[M]) -> M:
    """Parse JSON bytes into the given message dataclass (unknown keys ignored)."""
    return cls.from_dict(decode_json(raw))
