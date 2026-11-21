"""
Optimus Cluster Cache Coordinator (Controller-Side).
Queries worker cache summaries and inventories, tracks cache revisions to avoid
redundant metadata queries, and calculates byte-weighted dataset locality scores.
"""

from dataclasses import dataclass, field
import logging
import threading
from typing import Dict, List, Optional, Set, Tuple

from optimus.controller.communication import WorkerClientManager
from optimus.dataset import DatasetManifest
from optimus.protocol.generated.messages_pb2 import (
    CachedFileEntry,
    CacheInventoryResponse,
    CacheSummaryResponse,
)

logger = logging.getLogger("optimus.controller.cache_registry")


@dataclass
class WorkerCacheRecord:
    hostname: str
    port: int
    worker_id: str
    dataset_id: str
    dataset_version: str
    revision: int = 0
    cached_bytes: int = 0
    cached_files_count: int = 0
    cached_paths: Set[str] = field(default_factory=set)


class ClusterCacheCoordinator:
    """Coordinates and caches inventory metadata from all worker nodes."""

    def __init__(self, client_manager: WorkerClientManager):
        self.client_manager = client_manager
        self._cache_records: Dict[Tuple[str, int, str, str], WorkerCacheRecord] = {}
        self._lock = threading.Lock()

    def get_summary(
        self, hostname: str, port: int, dataset_id: str, dataset_version: str
    ) -> CacheSummaryResponse:
        """Queries a worker for a lightweight cache summary."""
        client = self.client_manager.get_client(hostname, port)
        return client.get_cache_summary(dataset_id, dataset_version)

    def get_inventory(
        self,
        hostname: str,
        port: int,
        dataset_id: str,
        dataset_version: str,
        force_refresh: bool = False,
    ) -> WorkerCacheRecord:
        """
        Retrieves cache inventory. If known revision matches worker revision,
        returns cached inventory immediately without transferring entire list.
        """
        key = (hostname, port, dataset_id, dataset_version)
        with self._lock:
            existing = self._cache_records.get(key)
            known_rev = 0 if force_refresh or not existing else existing.revision

        client = self.client_manager.get_client(hostname, port)
        page = 1
        page_size = 500
        all_files: List[CachedFileEntry] = []
        worker_id = f"worker-{hostname}"
        final_rev = 0
        total_cached_bytes = 0

        while True:
            resp: CacheInventoryResponse = client.get_cache_inventory(
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                known_revision=known_rev,
                page=page,
                page_size=page_size,
            )
            worker_id = resp.worker_id
            final_rev = resp.registry_revision
            total_cached_bytes = resp.total_cached_bytes

            if resp.revision_matched and existing is not None:
                logger.debug(
                    f"Cache revision {known_rev} matched for {hostname}:{port} ({dataset_id}:{dataset_version}). "
                    "Skipping inventory download."
                )
                return existing

            all_files.extend(resp.files)
            if page >= resp.total_pages or resp.total_pages == 0:
                break
            page += 1

        cached_paths = {f.relative_path for f in all_files if f.valid}
        record = WorkerCacheRecord(
            hostname=hostname,
            port=port,
            worker_id=worker_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            revision=final_rev,
            cached_bytes=total_cached_bytes,
            cached_files_count=len(cached_paths),
            cached_paths=cached_paths,
        )

        with self._lock:
            self._cache_records[key] = record

        return record

    def compute_locality(
        self,
        hostname: str,
        port: int,
        manifest: DatasetManifest,
    ) -> Tuple[float, int]:
        """
        Calculates byte-weighted dataset locality in [0.0, 1.0] and total cached bytes.
        locality = cached_required_dataset_bytes / total_required_dataset_bytes
        """
        if manifest.total_bytes == 0:
            return 1.0, 0

        record = self.get_inventory(
            hostname, port, manifest.dataset_id, manifest.version
        )

        cached_required_bytes = 0
        for entry in manifest.files:
            if entry.path in record.cached_paths:
                cached_required_bytes += entry.size

        locality = min(1.0, max(0.0, cached_required_bytes / float(manifest.total_bytes)))
        return locality, cached_required_bytes
