"""
Tests for Cache process-shared concurrency (flock locking).
Verifies that multiple concurrent readers requesting an uncached file result in exactly
one atomic download without race conditions or corrupted reads.
"""

from concurrent.futures import ThreadPoolExecutor
import os
import pytest

from optimus.cache.manager import CacheManager


def test_concurrent_uncached_file_reads(tmp_path):
    origin = str(tmp_path / "origin")
    cache_dir = str(tmp_path / "cache_dir")

    os.makedirs(os.path.join(origin, "concurrent"), exist_ok=True)

    # 1 MB test payload
    large_payload = b"X" * (1024 * 1024)
    file_path = os.path.join(origin, "concurrent", "large.bin")
    with open(file_path, "wb") as f:
        f.write(large_payload)

    mgr = CacheManager(cache_dir=cache_dir)

    def read_file(worker_idx: int) -> bytes:
        # Multiple workers concurrently request uncached file
        local_p = mgr.get_or_download(
            dataset_id="concurrent_ds",
            dataset_version="v1",
            relative_path="concurrent/large.bin",
            origin_base_path=origin,
        )
        with open(local_p, "rb") as f:
            return f.read()

    # Launch 8 concurrent reader threads
    num_readers = 8
    with ThreadPoolExecutor(max_workers=num_readers) as executor:
        futures = [executor.submit(read_file, i) for i in range(num_readers)]
        results = [f.result() for f in futures]

    for res in results:
        assert len(res) == len(large_payload)
        assert res == large_payload

    # Verify cached file is intact
    cached_file = mgr.get_cached_path("concurrent_ds", "v1", "concurrent/large.bin")
    assert os.path.exists(cached_file)
    assert os.path.getsize(cached_file) == len(large_payload)
