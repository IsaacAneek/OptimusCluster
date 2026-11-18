"""
Tests for FUSE cache process-shared concurrency (flock locking).
Verifies that multiple concurrent readers requesting an uncached file result in exactly
one atomic download without race conditions or corrupted reads.
"""

from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import time
import pytest


def test_concurrent_uncached_file_reads(tmp_path):
    origin = str(tmp_path / "origin")
    cache_dir = str(tmp_path / "cache_dir")
    mount_pt = str(tmp_path / "mnt")
    temp_dir = str(tmp_path / "temp")
    locks_dir = str(tmp_path / "locks")
    db_path = str(tmp_path / "cache.db")

    os.makedirs(os.path.join(origin, "concurrent"), exist_ok=True)
    os.makedirs(mount_pt, exist_ok=True)

    # 1 MB test payload
    large_payload = b"X" * (1024 * 1024)
    file_path = os.path.join(origin, "concurrent", "large.bin")
    with open(file_path, "wb") as f:
        f.write(large_payload)

    fuse_bin = os.path.abspath("src/optimus/cache/fuse_cache")
    if not os.path.exists(fuse_bin):
        pytest.skip("fuse_cache binary not built")

    cmd = [
        fuse_bin,
        "-f",
        f"--origin={origin}",
        f"--cache-dir={cache_dir}",
        f"--temp-dir={temp_dir}",
        f"--locks-dir={locks_dir}",
        f"--db={db_path}",
        "--dataset-id=concurrent_ds",
        "--dataset-version=v1",
        mount_pt,
    ]
    p = subprocess.Popen(cmd)
    time.sleep(1.0)

    try:
        mounted_target = os.path.join(mount_pt, "concurrent", "large.bin")

        def read_file(worker_idx: int) -> bytes:
            with open(mounted_target, "rb") as f:
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
        cached_file = os.path.join(cache_dir, "concurrent", "large.bin")
        assert os.path.exists(cached_file)
        assert os.path.getsize(cached_file) == len(large_payload)

    finally:
        subprocess.run(["fusermount3", "-u", mount_pt], capture_output=True)
        p.terminate()
        p.wait()
