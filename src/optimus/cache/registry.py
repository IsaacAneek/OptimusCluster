"""
Worker SQLite Cache Registry.
Tracks complete cached files, verification hashes, revisions, and access statistics.
Operates with WAL mode and process-safe transactions.
"""

from dataclasses import dataclass
import logging
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("optimus.cache.registry")

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS cached_files (
    dataset_id TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    local_path TEXT NOT NULL,
    valid INTEGER NOT NULL,
    last_access_time REAL,
    access_count INTEGER DEFAULT 0,
    download_time REAL,
    PRIMARY KEY (
        dataset_id,
        dataset_version,
        relative_path
    )
);

CREATE TABLE IF NOT EXISTS cache_statistics (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cache_hits INTEGER DEFAULT 0,
    cache_misses INTEGER DEFAULT 0,
    downloaded_bytes INTEGER DEFAULT 0,
    download_failures INTEGER DEFAULT 0,
    cached_bytes INTEGER DEFAULT 0,
    cached_files INTEGER DEFAULT 0,
    registry_revision INTEGER DEFAULT 0
);

INSERT OR IGNORE INTO cache_statistics (id, cache_hits, cache_misses, downloaded_bytes, download_failures, cached_bytes, cached_files, registry_revision)
VALUES (1, 0, 0, 0, 0, 0, 0, 0);
"""


@dataclass
class CachedFileRecord:
    dataset_id: str
    dataset_version: str
    relative_path: str
    file_size: int
    sha256: str
    local_path: str
    valid: bool
    last_access_time: float
    access_count: int
    download_time: float


class CacheRegistry:
    """Manages the worker's local SQLite cache database."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path:
            self.db_path = os.path.abspath(db_path)
        else:
            default_db = "/var/cache/optimus/cache.db"
            try:
                os.makedirs(os.path.dirname(default_db), exist_ok=True)
                self.db_path = default_db
            except (PermissionError, OSError):
                user_dir = os.path.expanduser("~/.cache/optimus")
                os.makedirs(user_dir, exist_ok=True)
                self.db_path = os.path.join(user_dir, "cache.db")

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA busy_timeout = 5000;")
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    def get_file(
        self, dataset_id: str, dataset_version: str, relative_path: str
    ) -> Optional[CachedFileRecord]:
        """Looks up a single cached file entry."""
        query = """
            SELECT dataset_id, dataset_version, relative_path, file_size, sha256,
                   local_path, valid, last_access_time, access_count, download_time
            FROM cached_files
            WHERE dataset_id = ? AND dataset_version = ? AND relative_path = ? AND valid = 1
        """
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(query, (dataset_id, dataset_version, relative_path))
            row = cur.fetchone()
            if not row:
                return None
            return CachedFileRecord(
                dataset_id=row["dataset_id"],
                dataset_version=row["dataset_version"],
                relative_path=row["relative_path"],
                file_size=row["file_size"],
                sha256=row["sha256"],
                local_path=row["local_path"],
                valid=bool(row["valid"]),
                last_access_time=row["last_access_time"],
                access_count=row["access_count"],
                download_time=row["download_time"],
            )

    def register_file(
        self,
        dataset_id: str,
        dataset_version: str,
        relative_path: str,
        file_size: int,
        sha256: str,
        local_path: str,
        download_time: float,
    ) -> None:
        """Atomically inserts or updates a valid cached file and bumps revision & statistics."""
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.cursor()
            # Insert or replace cached_files
            cur.execute(
                """
                INSERT INTO cached_files (
                    dataset_id, dataset_version, relative_path, file_size, sha256,
                    local_path, valid, last_access_time, access_count, download_time
                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?, 1, ?)
                ON CONFLICT(dataset_id, dataset_version, relative_path) DO UPDATE SET
                    file_size = excluded.file_size,
                    sha256 = excluded.sha256,
                    local_path = excluded.local_path,
                    valid = 1,
                    last_access_time = excluded.last_access_time,
                    access_count = access_count + 1,
                    download_time = excluded.download_time
                """,
                (
                    dataset_id,
                    dataset_version,
                    relative_path,
                    file_size,
                    sha256,
                    local_path,
                    now,
                    download_time,
                ),
            )

            # Update cache statistics
            cur.execute(
                """
                UPDATE cache_statistics SET
                    cache_misses = cache_misses + 1,
                    downloaded_bytes = downloaded_bytes + ?,
                    cached_bytes = (SELECT COALESCE(SUM(file_size), 0) FROM cached_files WHERE valid = 1),
                    cached_files = (SELECT COUNT(*) FROM cached_files WHERE valid = 1),
                    registry_revision = registry_revision + 1
                WHERE id = 1
                """,
                (file_size,),
            )
            conn.commit()

    def record_cache_hit(
        self, dataset_id: str, dataset_version: str, relative_path: str
    ) -> None:
        """Records a cache hit on an already-cached file."""
        now = time.time()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE cached_files SET
                    last_access_time = ?,
                    access_count = access_count + 1
                WHERE dataset_id = ? AND dataset_version = ? AND relative_path = ?
                """,
                (now, dataset_id, dataset_version, relative_path),
            )
            cur.execute(
                """
                UPDATE cache_statistics SET
                    cache_hits = cache_hits + 1
                WHERE id = 1
                """
            )
            conn.commit()

    def record_download_failure(self) -> None:
        """Records a failed download attempt."""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE cache_statistics SET download_failures = download_failures + 1 WHERE id = 1"
            )
            conn.commit()

    def get_summary(
        self, dataset_id: Optional[str] = None, dataset_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """Returns cache summary statistics and current registry revision."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM cache_statistics WHERE id = 1")
            stat_row = cur.fetchone()

            if dataset_id and dataset_version:
                cur.execute(
                    """
                    SELECT COUNT(*) as count, COALESCE(SUM(file_size), 0) as total_bytes
                    FROM cached_files
                    WHERE dataset_id = ? AND dataset_version = ? AND valid = 1
                    """,
                    (dataset_id, dataset_version),
                )
                ds_row = cur.fetchone()
                cached_files = ds_row["count"]
                cached_bytes = ds_row["total_bytes"]
            else:
                cached_files = stat_row["cached_files"]
                cached_bytes = stat_row["cached_bytes"]

            return {
                "cache_hits": stat_row["cache_hits"],
                "cache_misses": stat_row["cache_misses"],
                "downloaded_bytes": stat_row["downloaded_bytes"],
                "download_failures": stat_row["download_failures"],
                "cached_bytes": cached_bytes,
                "cached_files": cached_files,
                "registry_revision": stat_row["registry_revision"],
            }

    def get_inventory(
        self,
        dataset_id: str,
        dataset_version: str,
        page_number: int = 1,
        page_size: int = 500,
    ) -> Tuple[List[CachedFileRecord], int, int]:
        """
        Returns a paginated list of valid cached file records for a dataset version.
        Returns: (records, total_files, total_pages)
        """
        offset = max(0, (page_number - 1) * page_size)
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT COUNT(*) as cnt
                FROM cached_files
                WHERE dataset_id = ? AND dataset_version = ? AND valid = 1
                """,
                (dataset_id, dataset_version),
            )
            total_files = cur.fetchone()["cnt"]
            total_pages = max(1, (total_files + page_size - 1) // page_size)

            cur.execute(
                """
                SELECT dataset_id, dataset_version, relative_path, file_size, sha256,
                       local_path, valid, last_access_time, access_count, download_time
                FROM cached_files
                WHERE dataset_id = ? AND dataset_version = ? AND valid = 1
                ORDER BY relative_path ASC
                LIMIT ? OFFSET ?
                """,
                (dataset_id, dataset_version, page_size, offset),
            )
            rows = cur.fetchall()
            records = [
                CachedFileRecord(
                    dataset_id=r["dataset_id"],
                    dataset_version=r["dataset_version"],
                    relative_path=r["relative_path"],
                    file_size=r["file_size"],
                    sha256=r["sha256"],
                    local_path=r["local_path"],
                    valid=bool(r["valid"]),
                    last_access_time=r["last_access_time"],
                    access_count=r["access_count"],
                    download_time=r["download_time"],
                )
                for r in rows
            ]
            return records, total_files, total_pages
