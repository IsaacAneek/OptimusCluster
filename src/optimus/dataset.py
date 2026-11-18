"""
Optimus Dataset Definition and Manifest Structures.
Represents immutable dataset versions and their cryptographic manifests.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FileEntry:
    path: str  # Relative path within the dataset version directory
    size: int  # Exact file size in bytes
    sha256: str  # Hexadecimal SHA-256 digest


@dataclass
class DatasetManifest:
    dataset_id: str
    version: str
    total_bytes: int = 0
    total_files: int = 0
    files: List[FileEntry] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "version": self.version,
            "total_bytes": self.total_bytes,
            "total_files": self.total_files,
            "files": [asdict(f) for f in self.files],
        }

    def save(self, filepath: str) -> None:
        """Atomically writes the manifest JSON."""
        temp_path = f"{filepath}.tmp.{os.getpid()}"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, filepath)

    @classmethod
    def load(cls, filepath: str) -> "DatasetManifest":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        files = [
            FileEntry(path=item["path"], size=item["size"], sha256=item["sha256"])
            for item in data.get("files", [])
        ]
        return cls(
            dataset_id=data["dataset_id"],
            version=data["version"],
            total_bytes=data.get("total_bytes", sum(f.size for f in files)),
            total_files=data.get("total_files", len(files)),
            files=files,
        )


def compute_sha256(filepath: str, chunk_size: int = 65536) -> str:
    """Computes SHA-256 digest of a file in streaming 64 KiB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


class Dataset:
    """
    Developer-facing Dataset representation in the Optimus SDK.
    """

    def __init__(
        self,
        source: str,
        dataset_id: str,
        version: str = "v1",
    ):
        self.source = source
        self.dataset_id = dataset_id
        self.version = version
        self.manifest: Optional[DatasetManifest] = None

    def __repr__(self) -> str:
        return f"Dataset(id='{self.dataset_id}', version='{self.version}', source='{self.source}')"
