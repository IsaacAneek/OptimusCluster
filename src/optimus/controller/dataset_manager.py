"""
Optimus Dataset Manager (Controller-Side).
Prepares authoritative datasets, computes cryptographic manifests,
and validates NFS exports and worker mounts.
"""

import logging
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from typing import Optional
import urllib.request
import zipfile

from optimus.dataset import Dataset, DatasetManifest, FileEntry, compute_sha256

logger = logging.getLogger("optimus.controller.dataset_manager")


class DatasetManager:
    """Manages authoritative storage and manifest generation on the controller."""

    def __init__(self, storage_root: str = "/srv/datasets"):
        self.storage_root = os.path.abspath(storage_root)
        os.makedirs(self.storage_root, exist_ok=True)

    def get_dataset_version_dir(self, dataset_id: str, version: str) -> str:
        return os.path.join(self.storage_root, dataset_id, version)

    def get_manifest_path(self, dataset_id: str, version: str) -> str:
        return os.path.join(self.get_dataset_version_dir(dataset_id, version), "manifest.json")

    def has_manifest(self, dataset_id: str, version: str) -> bool:
        return os.path.isfile(self.get_manifest_path(dataset_id, version))

    def load_manifest(self, dataset_id: str, version: str) -> Optional[DatasetManifest]:
        manifest_file = self.get_manifest_path(dataset_id, version)
        if not os.path.isfile(manifest_file):
            return None
        return DatasetManifest.load(manifest_file)

    def prepare_dataset(self, dataset: Dataset, force_rehash: bool = False) -> DatasetManifest:
        """
        Imports or downloads dataset files into /storage_root/<id>/<version>/,
        computes SHA-256 for all files, and publishes manifest.json.
        """
        dest_dir = self.get_dataset_version_dir(dataset.dataset_id, dataset.version)
        os.makedirs(dest_dir, exist_ok=True)
        manifest_path = self.get_manifest_path(dataset.dataset_id, dataset.version)

        if not force_rehash and os.path.isfile(manifest_path):
            logger.info(f"Using existing manifest for {dataset.dataset_id}:{dataset.version}")
            manifest = DatasetManifest.load(manifest_path)
            dataset.manifest = manifest
            return manifest

        # Import source if not already populated
        source = dataset.source
        if os.path.isdir(source):
            if os.path.abspath(source) != os.path.abspath(dest_dir):
                logger.info(f"Copying dataset from directory '{source}' to '{dest_dir}'")
                for item in os.listdir(source):
                    s = os.path.join(source, item)
                    d = os.path.join(dest_dir, item)
                    if os.path.isdir(s):
                        if os.path.exists(d):
                            shutil.rmtree(d)
                        shutil.copytree(s, d)
                    else:
                        shutil.copy2(s, d)
        elif os.path.isfile(source):
            # Archive file
            self._extract_archive(source, dest_dir)
        elif source.startswith(("http://", "https://")):
            # Remote URL download
            temp_archive = os.path.join(dest_dir, f"download_{os.getpid()}.archive")
            try:
                logger.info(f"Downloading dataset from '{source}'...")
                urllib.request.urlretrieve(source, temp_archive)
                self._extract_archive(temp_archive, dest_dir)
            finally:
                if os.path.exists(temp_archive):
                    os.remove(temp_archive)

        # Generate manifest
        logger.info(f"Generating cryptographic manifest for {dataset.dataset_id}:{dataset.version}...")
        files: list[FileEntry] = []
        total_bytes = 0

        for root, _, filenames in os.walk(dest_dir):
            for fname in sorted(filenames):
                if fname == "manifest.json" or fname.startswith("."):
                    continue
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, dest_dir)
                size = os.path.getsize(full_path)
                sha = compute_sha256(full_path)
                files.append(FileEntry(path=rel_path, size=size, sha256=sha))
                total_bytes += size

        manifest = DatasetManifest(
            dataset_id=dataset.dataset_id,
            version=dataset.version,
            total_bytes=total_bytes,
            total_files=len(files),
            files=files,
        )
        manifest.save(manifest_path)
        logger.info(
            f"Published manifest: {len(files)} files, {total_bytes} bytes for {dataset.dataset_id}:{dataset.version}"
        )
        dataset.manifest = manifest
        return manifest

    def _extract_archive(self, archive_path: str, target_dir: str) -> None:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, "r") as z:
                z.extractall(target_dir)
        elif tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, "r:*") as t:
                t.extractall(target_dir)
        else:
            shutil.copy2(archive_path, target_dir)

    def verify_nfs_export(self, export_path: str = "/srv/datasets") -> bool:
        """Verifies if the specified path is exported via NFS on the controller."""
        target = os.path.abspath(export_path)
        # Check /var/lib/nfs/etab or exportfs output
        try:
            result = subprocess.run(["exportfs", "-v"], capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if target in line:
                        return True
        except Exception:
            pass

        # Check /etc/exports
        if os.path.isfile("/etc/exports"):
            try:
                with open("/etc/exports", "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and target in line:
                            return True
            except Exception:
                pass

        return False


def verify_worker_nfs_mount(mount_point: str = "/mnt/nfs-origin") -> bool:
    """
    Checks /proc/mounts to verify that the path is actually mounted as NFS (nfs or nfs4).
    Prevents mistaking an ordinary local directory for an NFS mount.
    """
    target = os.path.abspath(mount_point)
    if not os.path.exists(target):
        return False

    try:
        with open("/proc/mounts", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3:
                    mnt, fstype = parts[1], parts[2]
                    if os.path.abspath(mnt) == target and fstype in ("nfs", "nfs4"):
                        return True
    except Exception as e:
        logger.debug(f"Failed to read /proc/mounts: {e}")

    return False
