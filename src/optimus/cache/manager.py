"""
Optimus Worker Cache Manager.
Manages cache reconciliation, disk storage directories, and cache cleanup.
"""

import glob
import logging
import os
import shutil
import time
from typing import Optional

from optimus.cache.registry import CacheRegistry
from optimus.protocol.generated.messages_pb2 import (
    CachedFileEntry,
    CacheInventoryResponse,
    CacheSummaryResponse,
)

logger = logging.getLogger("optimus.cache.manager")


class CacheManager:
    """Manages local cache directories, startup reconciliation, and protocol conversion."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        db_path: Optional[str] = None,
        max_capacity_gb: int = 50,
    ):
        if cache_dir:
            self.cache_dir = os.path.abspath(cache_dir)
        else:
            default_dir = "/var/cache/optimus"
            try:
                os.makedirs(default_dir, exist_ok=True)
                self.cache_dir = default_dir
            except (PermissionError, OSError):
                self.cache_dir = os.path.expanduser("~/.cache/optimus")

        self.files_dir = os.path.join(self.cache_dir, "files")
        self.temp_dir = os.path.join(self.cache_dir, "temporary")
        self.locks_dir = os.path.join(self.cache_dir, "locks")
        self.max_capacity_gb = max_capacity_gb

        os.makedirs(self.files_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.locks_dir, exist_ok=True)

        self.db_path = db_path or os.path.join(self.cache_dir, "cache.db")
        self.registry = CacheRegistry(self.db_path)

        # Run startup reconciliation
        self.reconcile()

    def reconcile(self) -> None:
        """
        Cleans up incomplete downloads from prior crashes and ensures
        SQLite database accurately reflects local storage state.
        """
        logger.info(f"Reconciling cache directory '{self.cache_dir}'...")
        # 1. Clean stale temporary download files
        stale_files = glob.glob(os.path.join(self.temp_dir, "*.downloading"))
        for f in stale_files:
            try:
                os.remove(f)
                logger.info(f"Removed stale temporary download: {f}")
            except Exception as e:
                logger.warning(f"Could not remove stale download {f}: {e}")

        # 2. Check cached files validity against disk
        summary = self.registry.get_summary()
        logger.info(
            f"Cache initialized with {summary['cached_files']} files ({summary['cached_bytes']} bytes), "
            f"Revision: {summary['registry_revision']}"
        )

    def build_summary_response(
        self, worker_id: str, dataset_id: str, dataset_version: str
    ) -> CacheSummaryResponse:
        summary = self.registry.get_summary(dataset_id, dataset_version)
        return CacheSummaryResponse(
            worker_id=worker_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            cached_files_count=summary["cached_files"],
            total_cached_bytes=summary["cached_bytes"],
            cache_hits=summary["cache_hits"],
            cache_misses=summary["cache_misses"],
            downloaded_bytes=summary["downloaded_bytes"],
            registry_revision=summary["registry_revision"],
        )

    def build_inventory_response(
        self,
        worker_id: str,
        dataset_id: str,
        dataset_version: str,
        known_revision: int = 0,
        page_number: int = 1,
        page_size: int = 500,
    ) -> CacheInventoryResponse:
        summary = self.registry.get_summary(dataset_id, dataset_version)
        current_rev = summary["registry_revision"]

        # Fast path: if controller has the exact same revision, return revision_matched=True
        if known_revision > 0 and known_revision == current_rev:
            return CacheInventoryResponse(
                worker_id=worker_id,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                registry_revision=current_rev,
                revision_matched=True,
                total_cached_bytes=summary["cached_bytes"],
                total_cached_files=summary["cached_files"],
            )

        records, total_files, total_pages = self.registry.get_inventory(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            page_number=page_number,
            page_size=page_size,
        )

        entries = [
            CachedFileEntry(
                relative_path=r.relative_path,
                file_size=r.file_size,
                sha256=r.sha256,
                valid=r.valid,
                last_access_time=r.last_access_time,
            )
            for r in records
        ]

        return CacheInventoryResponse(
            worker_id=worker_id,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            registry_revision=current_rev,
            revision_matched=False,
            files=entries,
            current_page=page_number,
            total_pages=total_pages,
            total_cached_bytes=summary["cached_bytes"],
            total_cached_files=summary["cached_files"],
        )
