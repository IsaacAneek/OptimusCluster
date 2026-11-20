"""
Optimus Protocol Messages.
Pure Python message models replacing Protocol Buffers.
Provides clean serialization, typed access, and backward compatibility.
"""

from dataclasses import asdict, dataclass, field
import json
import time
from typing import Any, Dict, List, Optional
import uuid


class WorkerState:
    UNKNOWN = 0
    AVAILABLE = 1
    BUSY = 2
    UNAVAILABLE = 3

    _NAMES = {
        0: "UNKNOWN",
        1: "AVAILABLE",
        2: "BUSY",
        3: "UNAVAILABLE",
    }
    _VALUES = {v: k for k, v in _NAMES.items()}

    @classmethod
    def Name(cls, val: int) -> str:
        return cls._NAMES.get(val, "UNKNOWN")

    @classmethod
    def Value(cls, name: str) -> int:
        return cls._VALUES.get(name, 0)


class JobState:
    JOB_UNKNOWN = 0
    JOB_PENDING = 1
    JOB_RUNNING = 2
    JOB_COMPLETED = 3
    JOB_FAILED = 4
    JOB_STOPPED = 5

    _NAMES = {
        0: "JOB_UNKNOWN",
        1: "JOB_PENDING",
        2: "JOB_RUNNING",
        3: "JOB_COMPLETED",
        4: "JOB_FAILED",
        5: "JOB_STOPPED",
    }
    _VALUES = {v: k for k, v in _NAMES.items()}

    @classmethod
    def Name(cls, val: int) -> str:
        return cls._NAMES.get(val, "JOB_UNKNOWN")

    @classmethod
    def Value(cls, name: str) -> int:
        return cls._VALUES.get(name, 0)


class BaseMessage:
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def CopyFrom(self, other):
        for k in self.__dataclass_fields__.keys():
            if hasattr(other, k):
                setattr(self, k, getattr(other, k))


@dataclass
class ThermalZone(BaseMessage):
    zone_name: str = ""
    temperature_c: float = 0.0


@dataclass
class StatusRequest(BaseMessage):
    probe_only: bool = True


@dataclass
class StatusResponse(BaseMessage):
    worker_id: str = ""
    hostname: str = ""
    protocol_version: int = 1
    status: int = WorkerState.AVAILABLE
    active_job_count: int = 0
    uptime_seconds: int = 0


@dataclass
class ResourceRequest(BaseMessage):
    detailed: bool = False


@dataclass
class ResourceResponse(BaseMessage):
    worker_id: str = ""
    hostname: str = ""
    status: int = WorkerState.AVAILABLE
    cpu_percent: float = 0.0
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    cache_disk_total_gb: int = 0
    cache_disk_available_gb: int = 0
    temperature_c: float = 0.0
    temperature_available: bool = False
    active_jobs: int = 0
    thermals: List[ThermalZone] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        thermals_data = data.pop("thermals", [])
        thermals = [
            ThermalZone(**t) if isinstance(t, dict) else t for t in thermals_data
        ]
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys and k != "thermals"}
        return cls(thermals=thermals, **filtered)


@dataclass
class CacheSummaryRequest(BaseMessage):
    dataset_id: str = ""
    dataset_version: str = ""


@dataclass
class CacheSummaryResponse(BaseMessage):
    worker_id: str = ""
    dataset_id: str = ""
    dataset_version: str = ""
    cached_files_count: int = 0
    total_cached_bytes: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    downloaded_bytes: int = 0
    registry_revision: int = 0


@dataclass
class CachedFileEntry(BaseMessage):
    relative_path: str = ""
    file_size: int = 0
    sha256: str = ""
    valid: bool = True
    last_access_time: float = 0.0


@dataclass
class CacheInventoryRequest(BaseMessage):
    dataset_id: str = ""
    dataset_version: str = ""
    known_revision: int = 0
    page_number: int = 1
    page_size: int = 100


@dataclass
class CacheInventoryResponse(BaseMessage):
    worker_id: str = ""
    dataset_id: str = ""
    dataset_version: str = ""
    registry_revision: int = 0
    revision_matched: bool = False
    files: List[CachedFileEntry] = field(default_factory=list)
    current_page: int = 1
    total_pages: int = 1
    total_cached_bytes: int = 0
    total_cached_files: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        files_data = data.pop("files", [])
        files = [
            CachedFileEntry(**f) if isinstance(f, dict) else f for f in files_data
        ]
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys and k != "files"}
        return cls(files=files, **filtered)


@dataclass
class JobSubmit(BaseMessage):
    job_id: str = ""
    dataset_id: str = ""
    dataset_version: str = ""
    rank: int = 0
    world_size: int = 1
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    checkpoint_dir: str = "/srv/checkpoints"
    training_entrypoint: str = "optimus.training.worker_entry"
    training_args_json: str = "{}"
    required_ram_mb: int = 512
    resume_from_checkpoint: bool = False
    resume_checkpoint_path: str = ""


@dataclass
class JobResponse(BaseMessage):
    job_id: str = ""
    accepted: bool = True
    rejection_reason: str = ""
    reserved_ram_mb: int = 0


@dataclass
class JobStatusRequest(BaseMessage):
    job_id: str = ""


@dataclass
class JobStatusResponse(BaseMessage):
    job_id: str = ""
    worker_id: str = ""
    rank: int = 0
    status: int = JobState.JOB_UNKNOWN
    epoch: int = 0
    global_step: int = 0
    error_message: str = ""
    loss: float = 0.0
    throughput_samples_per_sec: float = 0.0


@dataclass
class JobStop(BaseMessage):
    job_id: str = ""
    reason: str = ""
    graceful: bool = True


@dataclass
class ShutdownRequest(BaseMessage):
    reason: str = ""
    force: bool = False


@dataclass
class ErrorResponse(BaseMessage):
    error_code: str = ""
    error_message: str = ""


_PAYLOAD_MAP = {
    "status_request": StatusRequest,
    "status_response": StatusResponse,
    "resource_request": ResourceRequest,
    "resource_response": ResourceResponse,
    "cache_summary_request": CacheSummaryRequest,
    "cache_summary_response": CacheSummaryResponse,
    "cache_inventory_request": CacheInventoryRequest,
    "cache_inventory_response": CacheInventoryResponse,
    "job_submit": JobSubmit,
    "job_response": JobResponse,
    "job_status_request": JobStatusRequest,
    "job_status_response": JobStatusResponse,
    "job_stop": JobStop,
    "shutdown_request": ShutdownRequest,
    "error_response": ErrorResponse,
}


class Envelope:
    """
    Unified communication envelope wrapping any typed payload.
    Supports attribute access (e.g. env.status_response) and WhichOneof().
    """

    def __init__(
        self,
        protocol_version: int = 1,
        message_id: Optional[str] = None,
        sender_id: str = "",
        timestamp_ms: Optional[int] = None,
        auth_token: str = "",
        payload_name: Optional[str] = None,
        payload_obj: Any = None,
    ):
        self.protocol_version = protocol_version
        self.message_id = message_id or str(uuid.uuid4())
        self.sender_id = sender_id
        self.timestamp_ms = timestamp_ms or int(time.time() * 1000)
        self.auth_token = auth_token
        self._payload_name = payload_name
        self._payload = payload_obj

        # Initialize slots for all known payloads to match protobuf attribute style
        for name in _PAYLOAD_MAP:
            setattr(self, name, None)

        if payload_name and payload_obj is not None:
            setattr(self, payload_name, payload_obj)

    def WhichOneof(self, oneof_group: str) -> Optional[str]:
        return self._payload_name

    def set_payload(self, name: str, obj: Any) -> None:
        self._payload_name = name
        self._payload = obj
        setattr(self, name, obj)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "protocol_version": self.protocol_version,
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "timestamp_ms": self.timestamp_ms,
            "auth_token": self.auth_token,
            "payload_name": self._payload_name,
            "payload": self._payload.to_dict() if hasattr(self._payload, "to_dict") else self._payload,
        }
        return d

    def SerializeToString(self) -> bytes:
        return json.dumps(self.to_dict()).encode("utf-8")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Envelope":
        payload_name = data.get("payload_name")
        payload_raw = data.get("payload")
        payload_obj = None

        if payload_name and payload_raw and payload_name in _PAYLOAD_MAP:
            cls_target = _PAYLOAD_MAP[payload_name]
            payload_obj = cls_target.from_dict(payload_raw)

        env = cls(
            protocol_version=data.get("protocol_version", 1),
            message_id=data.get("message_id"),
            sender_id=data.get("sender_id", ""),
            timestamp_ms=data.get("timestamp_ms"),
            auth_token=data.get("auth_token", ""),
            payload_name=payload_name,
            payload_obj=payload_obj,
        )
        return env

    @classmethod
    def FromString(cls, byte_data: bytes) -> "Envelope":
        parsed = json.loads(byte_data.decode("utf-8"))
        return cls.from_dict(parsed)
