"""
Optimus Protocol Package.
Provides the JSON message models and REST API definitions used between the
controller and optimusd worker daemons.
"""

from optimus.protocol.messages import (
    WorkerState,
    JobState,
    ThermalZone,
    StatusRequest,
    StatusResponse,
    ResourceRequest,
    ResourceResponse,
    CacheSummaryRequest,
    CacheSummaryResponse,
    CacheInventoryRequest,
    CacheInventoryResponse,
    CachedFileEntry,
    CheckpointEntry,
    CheckpointListRequest,
    CheckpointListResponse,
    JobSubmit,
    JobResponse,
    JobStatusRequest,
    JobStatusResponse,
    JobStop,
    ShutdownRequest,
    ErrorResponse,
)
from optimus.protocol.rest import (
    API_PREFIX,
    CURRENT_PROTOCOL_VERSION,
    DEFAULT_MAX_MESSAGE_SIZE,
    MessageTooLargeError,
    ProtocolError,
)

__all__ = [
    "WorkerState",
    "JobState",
    "ThermalZone",
    "StatusRequest",
    "StatusResponse",
    "ResourceRequest",
    "ResourceResponse",
    "CacheSummaryRequest",
    "CacheSummaryResponse",
    "CacheInventoryRequest",
    "CacheInventoryResponse",
    "CachedFileEntry",
    "CheckpointEntry",
    "CheckpointListRequest",
    "CheckpointListResponse",
    "JobSubmit",
    "JobResponse",
    "JobStatusRequest",
    "JobStatusResponse",
    "JobStop",
    "ShutdownRequest",
    "ErrorResponse",
    "API_PREFIX",
    "CURRENT_PROTOCOL_VERSION",
    "DEFAULT_MAX_MESSAGE_SIZE",
    "MessageTooLargeError",
    "ProtocolError",
]
