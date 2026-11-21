"""
Optimus Protocol Messages.
Pure Python dataclass models used as the JSON request/response bodies of the
optimusd REST API (see optimus.protocol.rest).
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


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


@dataclass
class CheckpointEntry(BaseMessage):
    job_id: str = ""
    checkpoint_name: str = ""
    epoch: int = 0
    global_step: int = 0
    loss: float = 0.0
    dataset_id: str = ""
    dataset_version: str = ""
    world_size: int = 1
    model_size_bytes: int = 0
    total_size_bytes: int = 0
    is_latest: bool = False
    saved_files: List[str] = field(default_factory=list)
    mtime: float = 0.0
    path: str = ""


@dataclass
class CheckpointListRequest(BaseMessage):
    checkpoint_dir: str = ""
    job_id: str = ""


@dataclass
class CheckpointListResponse(BaseMessage):
    worker_id: str = ""
    hostname: str = ""
    checkpoint_dir: str = ""
    accessible: bool = True
    disk_available_bytes: int = 0
    checkpoints: List[CheckpointEntry] = field(default_factory=list)
    error_message: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        ckpts_data = data.pop("checkpoints", [])
        checkpoints = [
            CheckpointEntry(**c) if isinstance(c, dict) else c for c in ckpts_data
        ]
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys and k != "checkpoints"}
        return cls(checkpoints=checkpoints, **filtered)
