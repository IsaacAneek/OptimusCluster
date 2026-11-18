"""
Worker Resource Information Provider.
Collects essential hardware telemetry: CPU utilization, RAM, cache disk space,
and CPU temperature following the 80/20 principle.
"""

import glob
import logging
import os
import shutil
from typing import Optional, Tuple
import psutil

from optimus.protocol.generated.messages_pb2 import (
    ResourceResponse,
    WorkerState,
)

logger = logging.getLogger("optimus.worker.resource_info")


class ResourceCollector:
    """Collects essential system resource metrics for scheduling."""

    def __init__(self, cache_dir: str = "/var/cache/optimus"):
        self.cache_dir = cache_dir
        # Prime psutil cpu_percent so consecutive calls return non-blocking diffs
        psutil.cpu_percent(interval=None)

    def get_cpu_percent(self) -> float:
        """Returns CPU utilization percentage across all cores."""
        try:
            return float(psutil.cpu_percent(interval=None))
        except Exception as e:
            logger.debug(f"Failed to read CPU percent: {e}")
            return 0.0

    def get_ram_mb(self) -> Tuple[int, int]:
        """Returns (total_ram_mb, available_ram_mb)."""
        try:
            mem = psutil.virtual_memory()
            total_mb = int(mem.total / (1024 * 1024))
            avail_mb = int(mem.available / (1024 * 1024))
            return total_mb, avail_mb
        except Exception as e:
            logger.debug(f"Failed to read RAM info: {e}")
            return 0, 0

    def get_cache_disk_gb(self) -> Tuple[int, int]:
        """Returns (total_disk_gb, available_disk_gb) for the cache directory."""
        target_dir = self.cache_dir
        # If cache directory doesn't exist yet, inspect its parent or root
        while target_dir and not os.path.exists(target_dir):
            parent = os.path.dirname(target_dir)
            if parent == target_dir:
                target_dir = "/"
                break
            target_dir = parent

        try:
            usage = shutil.disk_usage(target_dir)
            total_gb = int(usage.total / (1024 * 1024 * 1024))
            avail_gb = int(usage.free / (1024 * 1024 * 1024))
            return total_gb, avail_gb
        except Exception as e:
            logger.debug(f"Failed to read disk usage: {e}")
            return 0, 0

    def get_cpu_temperature(self) -> Tuple[Optional[float], bool]:
        """
        Reads CPU temperature in degrees Celsius from /sys/class/thermal or psutil.
        Returns (temperature_c, is_available).
        """
        # Try reading standard Linux thermal zones first (common on Raspberry Pi 5)
        thermal_zones = glob.glob("/sys/class/thermal/thermal_zone*/temp")
        for zone in thermal_zones:
            try:
                with open(zone, "r") as f:
                    content = f.read().strip()
                    if content.isdigit():
                        temp_c = float(content) / 1000.0
                        return temp_c, True
            except Exception:
                continue

        # Fallback to psutil sensors_temperatures
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current is not None:
                                return float(entry.current), True
        except Exception:
            pass

        return 0.0, False

    def collect(
        self,
        worker_id: str,
        hostname: str,
        current_state: WorkerState,
        active_jobs: int,
    ) -> ResourceResponse:
        """Assembles a full ResourceResponse message."""
        cpu = self.get_cpu_percent()
        ram_total, ram_avail = self.get_ram_mb()
        disk_total, disk_avail = self.get_cache_disk_gb()
        temp_c, temp_avail = self.get_cpu_temperature()

        status = current_state if active_jobs == 0 else WorkerState.BUSY

        return ResourceResponse(
            worker_id=worker_id,
            hostname=hostname,
            status=status,
            cpu_percent=cpu,
            ram_total_mb=ram_total,
            ram_available_mb=ram_avail,
            cache_disk_total_gb=disk_total,
            cache_disk_available_gb=disk_avail,
            temperature_c=temp_c,
            temperature_available=temp_avail,
            active_jobs=active_jobs,
        )
