"""
Optimus Worker Cache Manager.
Pure-Python atomic file caching with SQLite WAL metadata and POSIX file locks.
Replaces the C FUSE filesystem with a lightweight, high-performance Python caching engine.
"""

import fcntl
import glob
import hashlib
import logging
import os
import shutil
import time
from typing import Optional

from optimus.cache.registry import CacheRegistry
from optimus.protocol.messages import (
    CachedFileEntry,
    CacheInventoryResponse,
    CacheSummaryResponse,
)

logger = logging.getLogger("optimus.cache.manager")


class CacheManager:
    """Manages local cache directories, atomic downloads, and SQLite registry."""

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
        """Cleans up incomplete downloads from prior crashes."""
        logger.info(f"Reconciling cache directory '{self.cache_dir}'...")
        stale_files = glob.glob(os.path.join(self.temp_dir, "*.downloading"))
        for f in stale_files:
            try:
                os.remove(f)
                logger.info(f"Removed stale temporary download: {f}")
            except Exception as e:
                logger.warning(f"Could not remove stale download {f}: {e}")

        summary = self.registry.get_summary()
        logger.info(
            f"Cache initialized with {summary['cached_files']} files ({summary['cached_bytes']} bytes), "
            f"Revision: {summary['registry_revision']}"
        )

    def get_cached_path(self, dataset_id: str, dataset_version: str, relative_path: str) -> str:
        """Returns the expected local cache filesystem path for a file."""
        return os.path.join(self.files_dir, dataset_id, dataset_version, relative_path)

    def is_cached(self, dataset_id: str, dataset_version: str, relative_path: str) -> bool:
        """Checks if a file is locally cached and marked valid in registry."""
        local_path = self.get_cached_path(dataset_id, dataset_version, relative_path)
        if not os.path.exists(local_path):
            return False
        rec = self.registry.get_file(dataset_id, dataset_version, relative_path)
        return rec is not None and rec.valid

    def get_or_download(
        self,
        dataset_id: str,
        dataset_version: str,
        relative_path: str,
        origin_base_path: str,
        expected_sha256: Optional[str] = None,
        expected_size: Optional[int] = None,
    ) -> str:
        """
        Retrieves a dataset file from local cache (cache hit).
        On cache miss, synchronizes via flock, streams from origin_base_path,
        verifies SHA-256 hash, and atomically commits to local cache storage.
        """
        target_path = self.get_cached_path(dataset_id, dataset_version, relative_path)

        # 1. Fast path: already cached and valid
        if os.path.exists(target_path):
            rec = self.registry.get_file(dataset_id, dataset_version, relative_path)
            if rec and rec.valid:
                self.registry.record_cache_hit(dataset_id, dataset_version, relative_path)
                return target_path

        # 2. Cache miss: serialize across concurrent processes using lockfile
        lock_name = hashlib.md5(f"{dataset_id}_{dataset_version}_{relative_path}".encode()).hexdigest()
        lock_file = os.path.join(self.locks_dir, f"{lock_name}.lock")

        with open(lock_file, "w") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                # Re-check under lock
                if os.path.exists(target_path):
                    rec = self.registry.get_file(dataset_id, dataset_version, relative_path)
                    if rec and rec.valid:
                        self.registry.record_cache_hit(dataset_id, dataset_version, relative_path)
                        return target_path

                # Source path at origin (e.g. NFS mount or local origin)
                origin_file = os.path.join(origin_base_path, relative_path)
                if not os.path.exists(origin_file):
                    raise FileNotFoundError(f"Dataset file '{relative_path}' not found at origin: {origin_file}")

                # Download into temporary location
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                temp_file = os.path.join(
                    self.temp_dir, f"{lock_name}_{os.getpid()}_{int(time.time()*1000)}.downloading"
                )

                t_start = time.time()
                hasher = hashlib.sha256()
                bytes_copied = 0

                with open(origin_file, "rb") as src, open(temp_file, "wb") as dst:
                    while chunk := src.read(65536):
                        hasher.update(chunk)
                        dst.write(chunk)
                        bytes_copied += len(chunk)

                computed_sha256 = hasher.hexdigest()

                if expected_sha256 and computed_sha256 != expected_sha256:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    raise ValueError(
                        f"Checksum mismatch for '{relative_path}': expected {expected_sha256}, got {computed_sha256}"
                    )

                # Atomic publication
                os.replace(temp_file, target_path)
                download_time = max(0.001, time.time() - t_start)

                # Commit metadata into SQLite registry
                self.registry.register_file(
                    dataset_id=dataset_id,
                    dataset_version=dataset_version,
                    relative_path=relative_path,
                    file_size=bytes_copied,
                    sha256=computed_sha256,
                    local_path=target_path,
                    download_time=download_time,
                )
                return target_path
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)

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
