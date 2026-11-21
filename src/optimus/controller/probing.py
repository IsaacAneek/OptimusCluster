"""
Optimus Controller Probing Manager.
Handles controller-initiated status and availability probing.
Workers do not send heartbeats; the controller periodically polls each node.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from optimus.controller.communication import WorkerClientManager
from optimus.protocol.messages import (
    ResourceResponse,
    StatusResponse,
    WorkerState,
)

logger = logging.getLogger("optimus.controller.probing")


@dataclass
class WorkerNodeState:
    hostname: str
    port: int
    worker_id: str = "unknown"
    state: WorkerState = WorkerState.UNKNOWN
    active_jobs: int = 0
    uptime_seconds: int = 0
    consecutive_failures: int = 0
    last_probe_time: float = 0.0
    last_success_time: float = 0.0
    last_error: Optional[str] = None
    # Resource metrics
    cpu_percent: float = 0.0
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    cache_disk_total_gb: int = 0
    cache_disk_available_gb: int = 0
    temperature_c: float = 0.0
    temperature_available: bool = False


class ProbeManager:
    """Orchestrates periodic and on-demand probing of workers."""

    def __init__(
        self,
        client_manager: WorkerClientManager,
        failure_threshold: int = 3,
        probe_interval: float = 5.0,
    ):
        self.client_manager = client_manager
        self.failure_threshold = failure_threshold
        self.probe_interval = probe_interval

        self._workers: Dict[Tuple[str, int], WorkerNodeState] = {}
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None

    def register_worker(
        self, hostname: str, port: int, worker_id: Optional[str] = None
    ) -> None:
        key = (hostname, port)
        with self._lock:
            if key not in self._workers:
                self._workers[key] = WorkerNodeState(
                    hostname=hostname,
                    port=port,
                    worker_id=worker_id or f"worker-{hostname}",
                    state=WorkerState.UNKNOWN,
                )

    def unregister_worker(self, hostname: str, port: int) -> None:
        key = (hostname, port)
        with self._lock:
            self._workers.pop(key, None)

    def probe_worker(self, hostname: str, port: int, probe_only: bool = False) -> WorkerNodeState:
        """
        Sends unified probe to a worker and updates availability and resource metrics.
        Consolidates telemetry into a single HTTP request.
        """
        key = (hostname, port)
        with self._lock:
            if key not in self._workers:
                self._workers[key] = WorkerNodeState(
                    hostname=hostname, port=port, state=WorkerState.UNKNOWN
                )
            node_state = self._workers[key]

        client = self.client_manager.get_client(hostname, port)
        now = time.time()
        node_state.last_probe_time = now

        try:
            if not probe_only:
                resp: ResourceResponse = client.get_resources()
                with self._lock:
                    node_state.worker_id = resp.worker_id
                    node_state.state = resp.status
                    node_state.active_jobs = resp.active_jobs
                    node_state.cpu_percent = resp.cpu_percent
                    node_state.ram_total_mb = resp.ram_total_mb
                    node_state.ram_available_mb = resp.ram_available_mb
                    node_state.cache_disk_total_gb = resp.cache_disk_total_gb
                    node_state.cache_disk_available_gb = resp.cache_disk_available_gb
                    node_state.temperature_c = resp.temperature_c
                    node_state.temperature_available = resp.temperature_available
                    node_state.consecutive_failures = 0
                    node_state.last_success_time = now
                    node_state.last_error = None
            else:
                resp: StatusResponse = client.get_status()
                with self._lock:
                    node_state.worker_id = resp.worker_id
                    node_state.state = resp.status
                    node_state.active_jobs = resp.active_job_count
                    node_state.uptime_seconds = resp.uptime_seconds
                    node_state.consecutive_failures = 0
                    node_state.last_success_time = now
                    node_state.last_error = None

        except Exception as ex:
            with self._lock:
                node_state.consecutive_failures += 1
                node_state.last_error = str(ex)
                if node_state.consecutive_failures >= self.failure_threshold:
                    node_state.state = WorkerState.UNAVAILABLE
                    logger.warning(
                        f"Worker {hostname}:{port} marked UNAVAILABLE after "
                        f"{node_state.consecutive_failures} failed probes. Last error: {ex}"
                    )
                else:
                    logger.debug(
                        f"Worker {hostname}:{port} probe failed ({node_state.consecutive_failures}/"
                        f"{self.failure_threshold}): {ex}"
                    )

        return node_state

    def probe_all(self) -> Dict[Tuple[str, int], WorkerNodeState]:
        """Probes all registered workers in parallel and returns updated states."""
        with self._lock:
            targets = list(self._workers.keys())

        if not targets:
            return {}

        results = {}
        with ThreadPoolExecutor(max_workers=min(32, len(targets))) as executor:
            future_to_worker = {
                executor.submit(self.probe_worker, host, port): (host, port)
                for host, port in targets
            }
            for future in future_to_worker:
                key = future_to_worker[future]
                try:
                    results[key] = future.result()
                except Exception as ex:
                    logger.error(f"Error probing {key}: {ex}")

        return results

    def get_all_states(self) -> List[WorkerNodeState]:
        with self._lock:
            return list(self._workers.values())

    def start_probing_loop(self) -> None:
        """Starts background periodic probing loop initiated by the controller."""
        if self._running.is_set():
            return
        self._running.set()
        self._loop_thread = threading.Thread(
            target=self._probing_loop, name="ControllerProbeLoop", daemon=True
        )
        self._loop_thread.start()
        logger.info(f"Controller probing loop started (interval: {self.probe_interval}s)")

    def _probing_loop(self) -> None:
        while self._running.is_set():
            start_t = time.time()
            try:
                self.probe_all()
            except Exception as e:
                logger.error(f"Error during scheduled probe: {e}")
            elapsed = time.time() - start_t
            sleep_time = max(0.1, self.probe_interval - elapsed)
            time.sleep(sleep_time)

    def stop_probing_loop(self) -> None:
        self._running.clear()
        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2.0)
        logger.info("Controller probing loop stopped")

    def probe_worker_resources(self, hostname: str, port: int) -> WorkerNodeState:
        """Unified probe returning node status and resources."""
        return self.probe_worker(hostname, port, probe_only=False)

    def probe_all_resources(self) -> Dict[Tuple[str, int], WorkerNodeState]:
        """Unified probe across all workers."""
        return self.probe_all()

    def format_worker_table(self) -> str:
        """Generates a text table of current worker status and resource metrics."""
        headers = (
            f"{'Worker ID':<20} {'Hostname':<18} {'Status':<12} {'CPU%':<7} "
            f"{'RAM (MB)':<14} {'Disk Avail':<11} {'Temp(C)':<8} {'Jobs':<5}"
        )
        lines = [headers, "-" * len(headers)]
        with self._lock:
            for s in self._workers.values():
                status_str = WorkerState.Name(s.state)
                ram_str = f"{s.ram_available_mb}/{s.ram_total_mb}" if s.ram_total_mb > 0 else "N/A"
                disk_str = f"{s.cache_disk_available_gb} GB" if s.cache_disk_total_gb > 0 else "N/A"
                temp_str = f"{s.temperature_c:.1f}" if s.temperature_available else "N/A"
                cpu_str = f"{s.cpu_percent:.1f}%"

                lines.append(
                    f"{s.worker_id:<20} {s.hostname:<18} {status_str:<12} {cpu_str:<7} "
                    f"{ram_str:<14} {disk_str:<11} {temp_str:<8} {s.active_jobs:<5}"
                )
        return "\n".join(lines)
