"""
Comprehensive tests for Optimus Cache subsystem (Phases 5 & 6):
- SQLite Cache Registry
- Worker Cache Manager & Protocol Conversion
- Controller Cache Inventory Coordinator & Locality Scoring
- Live FUSE Caching (Miss, Hit, and Zero-Origin Verification)
"""

import os
import shutil
import socket
import subprocess
import tempfile
import time
import pytest

from optimus.cache.manager import CacheManager
from optimus.cache.registry import CacheRegistry
from optimus.controller.cache_registry import ClusterCacheCoordinator
from optimus.controller.communication import WorkerClientManager
from optimus.dataset import DatasetManifest, FileEntry
from optimus.worker.server import WorkerServer


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_sqlite_cache_registry(tmp_path):
    db_file = str(tmp_path / "cache.db")
    reg = CacheRegistry(db_file)

    # Initial summary
    summary = reg.get_summary()
    assert summary["cached_files"] == 0
    assert summary["cached_bytes"] == 0
    assert summary["registry_revision"] == 0

    # Register first file
    reg.register_file(
        dataset_id="animals",
        dataset_version="v1",
        relative_path="cat/01.jpg",
        file_size=2048,
        sha256="abc123sha",
        local_path="/cache/cat/01.jpg",
        download_time=0.05,
    )

    file_rec = reg.get_file("animals", "v1", "cat/01.jpg")
    assert file_rec is not None
    assert file_rec.file_size == 2048
    assert file_rec.sha256 == "abc123sha"
    assert file_rec.valid is True
    assert file_rec.access_count == 1

    summary = reg.get_summary("animals", "v1")
    assert summary["cached_files"] == 1
    assert summary["cached_bytes"] == 2048
    assert summary["registry_revision"] == 1

    # Record cache hit
    reg.record_cache_hit("animals", "v1", "cat/01.jpg")
    file_rec = reg.get_file("animals", "v1", "cat/01.jpg")
    assert file_rec.access_count == 2
    summary = reg.get_summary()
    assert summary["cache_hits"] == 1

    # Inventory pagination
    records, total_files, total_pages = reg.get_inventory("animals", "v1", page_number=1, page_size=10)
    assert total_files == 1
    assert total_pages == 1
    assert len(records) == 1
    assert records[0].relative_path == "cat/01.jpg"


def test_cache_inventory_probing_and_locality(tmp_path):
    cache_dir = str(tmp_path / "cache")
    port = get_free_port()

    # Pre-populate cache with one file
    mgr = CacheManager(cache_dir=cache_dir)
    mgr.registry.register_file(
        dataset_id="animals",
        dataset_version="v1",
        relative_path="dog/01.jpg",
        file_size=5000,
        sha256="doghash",
        local_path=os.path.join(cache_dir, "dog/01.jpg"),
        download_time=0.1,
    )

    worker = WorkerServer(
        worker_id="pi-cache-01",
        hostname="localhost",
        port=port,
        cache_dir=cache_dir,
    )
    worker.start()

    client_mgr = WorkerClientManager(default_timeout=2.0)
    coord = ClusterCacheCoordinator(client_mgr)

    try:
        # Summary query
        summary_resp = coord.get_summary("localhost", port, "animals", "v1")
        assert summary_resp.cached_files_count == 1
        assert summary_resp.total_cached_bytes == 5000
        assert summary_resp.registry_revision >= 1

        # Inventory query
        inv_record = coord.get_inventory("localhost", port, "animals", "v1")
        assert inv_record.cached_files_count == 1
        assert "dog/01.jpg" in inv_record.cached_paths

        # Verify revision caching: calling again returns cached record without re-querying all files
        inv_record2 = coord.get_inventory("localhost", port, "animals", "v1")
        assert inv_record2 is inv_record

        # Locality calculation
        manifest = DatasetManifest(
            dataset_id="animals",
            version="v1",
            total_bytes=10000,
            total_files=2,
            files=[
                FileEntry(path="dog/01.jpg", size=5000, sha256="doghash"),
                FileEntry(path="cat/02.jpg", size=5000, sha256="cathash"),
            ],
        )
        locality, cached_bytes = coord.compute_locality("localhost", port, manifest)
        assert locality == 0.5
        assert cached_bytes == 5000

    finally:
        worker.stop()
        client_mgr.close_all()


def test_live_fuse_cache_miss_and_hit(tmp_path):
    origin = str(tmp_path / "origin")
    cache_dir = str(tmp_path / "cache_dir")
    mount_pt = str(tmp_path / "mnt")
    temp_dir = str(tmp_path / "temp")
    locks_dir = str(tmp_path / "locks")
    db_path = str(tmp_path / "cache.db")

    os.makedirs(os.path.join(origin, "sub"), exist_ok=True)
    os.makedirs(mount_pt, exist_ok=True)

    test_content = b"PyTorch Distributed Caching Data Payload"
    with open(os.path.join(origin, "sub", "tensor.bin"), "wb") as f:
        f.write(test_content)

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
        "--dataset-id=tensors",
        "--dataset-version=v1",
        mount_pt,
    ]
    p = subprocess.Popen(cmd)
    time.sleep(1.0)

    try:
        mf = os.path.join(mount_pt, "sub", "tensor.bin")
        # Cold read (Cache Miss)
        with open(mf, "rb") as f:
            read_cold = f.read()
        assert read_cold == test_content

        # Verify file is stored in local cache
        cached_local = os.path.join(cache_dir, "sub", "tensor.bin")
        assert os.path.exists(cached_local)

        # Remove origin file to guarantee warm read does not touch origin
        os.remove(os.path.join(origin, "sub", "tensor.bin"))

        # Warm read (Cache Hit)
        with open(mf, "rb") as f:
            read_warm = f.read()
        assert read_warm == test_content

    finally:
        subprocess.run(["fusermount3", "-u", mount_pt], capture_output=True)
        p.terminate()
        p.wait()
