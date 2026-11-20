"""
Optimus Worker TCP Server.
Listens for controller requests, validates authentication, and dispatches responses.
Workers NEVER push heartbeats to the controller; they only respond when queried.
"""

import json
import logging
import os
import shutil
import socket
import threading
import time
from typing import Callable, Dict, List, Optional

from optimus.cache.manager import CacheManager
from optimus.protocol.framing import (
    CURRENT_PROTOCOL_VERSION,
    DEFAULT_MAX_MESSAGE_SIZE,
    create_envelope,
    recv_message,
    send_message,
    ProtocolError,
)
from optimus.protocol.generated.messages_pb2 import (
    CacheInventoryRequest,
    CacheInventoryResponse,
    CacheSummaryRequest,
    CacheSummaryResponse,
    CheckpointEntry,
    CheckpointListRequest,
    CheckpointListResponse,
    Envelope,
    ErrorResponse,
    JobResponse,
    JobState,
    JobStatusRequest,
    JobStatusResponse,
    JobStop,
    JobSubmit,
    ResourceResponse,
    StatusResponse,
    WorkerState,
)
from optimus.worker.executor import TrainingExecutor
from optimus.worker.resource_info import ResourceCollector

logger = logging.getLogger("optimus.worker.server")


class WorkerServer:
    def __init__(
        self,
        worker_id: str,
        hostname: str,
        port: int = 45870,
        host: str = "0.0.0.0",
        cluster_secret: str = "",
        max_message_size: int = DEFAULT_MAX_MESSAGE_SIZE,
        cache_dir: Optional[str] = None,
        logs_dir: str = "./logs",
    ):
        self.worker_id = worker_id
        self.hostname = hostname
        self.port = port
        self.host = host
        self.cluster_secret = cluster_secret
        self.max_message_size = max_message_size

        if cache_dir:
            self.cache_dir = cache_dir
        else:
            # Check if /var/cache/optimus is accessible, otherwise use user cache
            default_dir = "/var/cache/optimus"
            try:
                os.makedirs(default_dir, exist_ok=True)
                self.cache_dir = default_dir
            except (PermissionError, OSError):
                self.cache_dir = os.path.expanduser("~/.cache/optimus")
                os.makedirs(self.cache_dir, exist_ok=True)

        self.logs_dir = logs_dir
        self.resource_collector = ResourceCollector(cache_dir=self.cache_dir)
        self.cache_manager = CacheManager(cache_dir=self.cache_dir)
        self.executor = TrainingExecutor(logs_dir=self.logs_dir)

        self.start_time = time.time()
        self.state = WorkerState.AVAILABLE
        self.active_jobs = 0

        # Dispatch table for payload handlers
        self._handlers: Dict[str, Callable[[Envelope], Envelope]] = {}
        self._register_default_handlers()

        self._server_sock: Optional[socket.socket] = None
        self._is_running = threading.Event()
        self._server_thread: Optional[threading.Thread] = None
        self._active_connections = set()
        self._lock = threading.Lock()

    def register_handler(
        self, payload_name: str, handler: Callable[[Envelope], Envelope]
    ) -> None:
        """Register custom payload handlers (used by resource, cache, job modules)."""
        self._handlers[payload_name] = handler

    def _register_default_handlers(self) -> None:
        self._handlers["status_request"] = self._handle_status_request
        self._handlers["resource_request"] = self._handle_resource_request
        self._handlers["cache_summary_request"] = self._handle_cache_summary_request
        self._handlers["cache_inventory_request"] = self._handle_cache_inventory_request
        self._handlers["job_submit"] = self._handle_job_submit
        self._handlers["job_status_request"] = self._handle_job_status_request
        self._handlers["job_stop"] = self._handle_job_stop
        self._handlers["shutdown_request"] = self._handle_shutdown_request
        self._handlers["checkpoint_list_request"] = self._handle_checkpoint_list_request

    def _handle_checkpoint_list_request(self, envelope: Envelope) -> Envelope:
        req: CheckpointListRequest = envelope.checkpoint_list_request
        target_dir = req.checkpoint_dir if req and req.checkpoint_dir else "/srv/checkpoints"
        job_filter = req.job_id if req and req.job_id else ""
        resp = self._scan_checkpoints(target_dir, job_filter)
        return create_envelope("checkpoint_list_response", resp, self.worker_id)

    def _scan_checkpoints(self, target_dir: str, job_filter: str = "") -> CheckpointListResponse:
        target_dir = os.path.abspath(target_dir)
        if not os.path.isdir(target_dir):
            return CheckpointListResponse(
                worker_id=self.worker_id,
                hostname=self.hostname,
                checkpoint_dir=target_dir,
                accessible=False,
                error_message=f"Directory '{target_dir}' does not exist on this node",
            )

        disk_avail = 0
        try:
            usage = shutil.disk_usage(target_dir)
            disk_avail = usage.free
        except Exception:
            pass

        checkpoints: List[CheckpointEntry] = []
        try:
            for job_name in os.listdir(target_dir):
                if job_filter and job_name != job_filter:
                    continue
                job_path = os.path.join(target_dir, job_name)
                if not os.path.isdir(job_path):
                    continue

                latest_target = ""
                latest_file = os.path.join(job_path, "latest.json")
                if os.path.isfile(latest_file):
                    try:
                        with open(latest_file, "r") as f:
                            ldata = json.load(f)
                        latest_target = os.path.basename(ldata.get("latest_checkpoint", ""))
                    except Exception:
                        pass

                for item in os.listdir(job_path):
                    if item.startswith("checkpoint-") and not item.endswith(".tmp"):
                        ckpt_dir = os.path.join(job_path, item)
                        if not os.path.isdir(ckpt_dir):
                            continue

                        meta_dict = {}
                        meta_path = os.path.join(ckpt_dir, "metadata.json")
                        if os.path.isfile(meta_path):
                            try:
                                with open(meta_path, "r") as f:
                                    meta_dict = json.load(f)
                            except Exception:
                                pass

                        files = []
                        total_bytes = 0
                        model_bytes = 0
                        mtime = 0.0
                        try:
                            for fname in os.listdir(ckpt_dir):
                                fpath = os.path.join(ckpt_dir, fname)
                                if os.path.isfile(fpath):
                                    files.append(fname)
                                    fsize = os.path.getsize(fpath)
                                    total_bytes += fsize
                                    if fname == "model.pt":
                                        model_bytes = fsize
                                    fmtime = os.path.getmtime(fpath)
                                    if fmtime > mtime:
                                        mtime = fmtime
                        except Exception:
                            pass

                        epoch = meta_dict.get("epoch", 0)
                        if "epoch" not in meta_dict and item.startswith("checkpoint-"):
                            try:
                                epoch = int(item.split("-")[1])
                            except Exception:
                                pass

                        entry = CheckpointEntry(
                            job_id=job_name,
                            checkpoint_name=item,
                            epoch=epoch,
                            global_step=meta_dict.get("global_step", 0),
                            loss=float(meta_dict.get("loss", 0.0)),
                            dataset_id=meta_dict.get("dataset_id", ""),
                            dataset_version=meta_dict.get("dataset_version", ""),
                            world_size=int(meta_dict.get("world_size", 1)),
                            model_size_bytes=model_bytes,
                            total_size_bytes=total_bytes,
                            is_latest=(item == latest_target),
                            saved_files=files,
                            mtime=mtime,
                            path=ckpt_dir,
                        )
                        checkpoints.append(entry)

            checkpoints.sort(key=lambda c: (c.job_id, c.epoch), reverse=True)
            return CheckpointListResponse(
                worker_id=self.worker_id,
                hostname=self.hostname,
                checkpoint_dir=target_dir,
                accessible=True,
                disk_available_bytes=disk_avail,
                checkpoints=checkpoints,
            )
        except Exception as e:
            return CheckpointListResponse(
                worker_id=self.worker_id,
                hostname=self.hostname,
                checkpoint_dir=target_dir,
                accessible=False,
                error_message=str(e),
            )

    def _handle_job_submit(self, envelope: Envelope) -> Envelope:
        req: JobSubmit = envelope.job_submit
        resp = self.executor.submit_job(req)
        if resp.accepted:
            self.active_jobs = 1
        return create_envelope("job_response", resp, self.worker_id)

    def _handle_job_status_request(self, envelope: Envelope) -> Envelope:
        req: JobStatusRequest = envelope.job_status_request
        resp = self.executor.get_status(req.job_id, self.worker_id)
        if resp.status in (JobState.JOB_COMPLETED, JobState.JOB_FAILED, JobState.JOB_STOPPED):
            self.active_jobs = 0
        return create_envelope("job_status_response", resp, self.worker_id)

    def _handle_job_stop(self, envelope: Envelope) -> Envelope:
        req: JobStop = envelope.job_stop
        self.executor.stop_job(req.job_id, req.reason)
        self.active_jobs = 0
        status_resp = StatusResponse(
            worker_id=self.worker_id,
            hostname=self.hostname,
            protocol_version=CURRENT_PROTOCOL_VERSION,
            status=WorkerState.AVAILABLE,
            active_job_count=0,
            uptime_seconds=int(time.time() - self.start_time),
        )
        return create_envelope("status_response", status_resp, self.worker_id)

    def _handle_cache_summary_request(self, envelope: Envelope) -> Envelope:
        req: CacheSummaryRequest = envelope.cache_summary_request
        resp = self.cache_manager.build_summary_response(
            worker_id=self.worker_id,
            dataset_id=req.dataset_id,
            dataset_version=req.dataset_version,
        )
        return create_envelope("cache_summary_response", resp, self.worker_id)

    def _handle_cache_inventory_request(self, envelope: Envelope) -> Envelope:
        req: CacheInventoryRequest = envelope.cache_inventory_request
        resp = self.cache_manager.build_inventory_response(
            worker_id=self.worker_id,
            dataset_id=req.dataset_id,
            dataset_version=req.dataset_version,
            known_revision=req.known_revision,
            page_number=req.page_number or 1,
            page_size=req.page_size or 500,
        )
        return create_envelope("cache_inventory_response", resp, self.worker_id)

    def _handle_resource_request(self, envelope: Envelope) -> Envelope:
        resp = self.resource_collector.collect(
            worker_id=self.worker_id,
            hostname=self.hostname,
            current_state=self.state,
            active_jobs=self.active_jobs,
        )
        return create_envelope(
            payload_field_name="resource_response",
            payload_obj=resp,
            sender_id=self.worker_id,
            protocol_version=CURRENT_PROTOCOL_VERSION,
        )

    def _handle_status_request(self, envelope: Envelope) -> Envelope:
        resp = StatusResponse()
        resp.worker_id = self.worker_id
        resp.hostname = self.hostname
        resp.protocol_version = CURRENT_PROTOCOL_VERSION
        resp.status = self.state if self.active_jobs == 0 else WorkerState.BUSY
        resp.active_job_count = self.active_jobs
        resp.uptime_seconds = int(time.time() - self.start_time)

        return create_envelope(
            payload_field_name="status_response",
            payload_obj=resp,
            sender_id=self.worker_id,
            protocol_version=CURRENT_PROTOCOL_VERSION,
        )

    def _handle_shutdown_request(self, envelope: Envelope) -> Envelope:
        logger.info(f"Shutdown requested by {envelope.sender_id}")
        threading.Thread(target=self.stop, daemon=True).start()
        resp = StatusResponse()
        resp.worker_id = self.worker_id
        resp.hostname = self.hostname
        resp.protocol_version = CURRENT_PROTOCOL_VERSION
        resp.status = WorkerState.UNAVAILABLE
        resp.active_job_count = self.active_jobs
        resp.uptime_seconds = int(time.time() - self.start_time)
        return create_envelope("status_response", resp, self.worker_id)

    def start(self) -> None:
        """Starts the TCP server in a background thread."""
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(128)
        self._is_running.set()

        self._server_thread = threading.Thread(
            target=self._accept_loop, name=f"WorkerServer-{self.port}", daemon=True
        )
        self._server_thread.start()
        logger.info(f"Worker TCP server listening on {self.host}:{self.port} (ID: {self.worker_id})")

    def _accept_loop(self) -> None:
        self._server_sock.settimeout(1.0)
        while self._is_running.is_set():
            try:
                client_sock, client_addr = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            t = threading.Thread(
                target=self._client_handler,
                args=(client_sock, client_addr),
                daemon=True,
            )
            with self._lock:
                self._active_connections.add(client_sock)
            t.start()

    def _client_handler(self, sock: socket.socket, addr) -> None:
        logger.debug(f"Accepted connection from {addr}")
        try:
            sock.settimeout(10.0)
            while self._is_running.is_set():
                try:
                    envelope = recv_message(
                        sock=sock,
                        max_message_size=self.max_message_size,
                        expected_protocol_version=None,
                    )
                except (ConnectionResetError, ConnectionError, OSError):
                    break
                except socket.timeout:
                    continue
                except ProtocolError as e:
                    logger.warning(f"Protocol error from {addr}: {e}")
                    err = ErrorResponse(
                        error_code="PROTOCOL_ERROR", error_message=str(e)
                    )
                    err_env = create_envelope(
                        "error_response", err, sender_id=self.worker_id
                    )
                    try:
                        send_message(sock, err_env, self.max_message_size)
                    except Exception:
                        pass
                    break

                # Validate Protocol Version
                if envelope.protocol_version != CURRENT_PROTOCOL_VERSION:
                    err = ErrorResponse(
                        error_code="VERSION_MISMATCH",
                        error_message=(
                            f"Supported protocol version is {CURRENT_PROTOCOL_VERSION}, "
                            f"received {envelope.protocol_version}"
                        ),
                    )
                    err_env = create_envelope(
                        "error_response", err, sender_id=self.worker_id
                    )
                    send_message(sock, err_env, self.max_message_size)
                    continue

                # Validate Auth Token if cluster secret is configured
                if (
                    self.cluster_secret
                    and envelope.auth_token != self.cluster_secret
                ):
                    logger.warning(f"Unauthorized request from {addr}")
                    err = ErrorResponse(
                        error_code="UNAUTHORIZED",
                        error_message="Invalid cluster authentication token",
                    )
                    err_env = create_envelope(
                        "error_response", err, sender_id=self.worker_id
                    )
                    send_message(sock, err_env, self.max_message_size)
                    continue

                payload_type = envelope.WhichOneof("payload")
                if not payload_type:
                    err = ErrorResponse(
                        error_code="EMPTY_PAYLOAD",
                        error_message="Received envelope with no payload",
                    )
                    err_env = create_envelope(
                        "error_response", err, sender_id=self.worker_id
                    )
                    send_message(sock, err_env, self.max_message_size)
                    continue

                handler = self._handlers.get(payload_type)
                if not handler:
                    err = ErrorResponse(
                        error_code="UNSUPPORTED_REQUEST",
                        error_message=f"No handler registered for {payload_type}",
                    )
                    err_env = create_envelope(
                        "error_response", err, sender_id=self.worker_id
                    )
                    send_message(sock, err_env, self.max_message_size)
                    continue

                try:
                    response_env = handler(envelope)
                    if response_env:
                        send_message(sock, response_env, self.max_message_size)
                except Exception as ex:
                    logger.exception(f"Error handling {payload_type}: {ex}")
                    err = ErrorResponse(
                        error_code="INTERNAL_ERROR", error_message=str(ex)
                    )
                    err_env = create_envelope(
                        "error_response", err, sender_id=self.worker_id
                    )
                    send_message(sock, err_env, self.max_message_size)

        finally:
            with self._lock:
                self._active_connections.discard(sock)
            try:
                sock.close()
            except Exception:
                pass
            logger.debug(f"Closed connection from {addr}")

    def stop(self) -> None:
        """Stops the worker TCP server and closes connections."""
        self._is_running.clear()
        if self._server_sock:
            try:
                self._server_sock.close()
            except Exception:
                pass
        with self._lock:
            for sock in list(self._active_connections):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                    sock.close()
                except Exception:
                    pass
            self._active_connections.clear()
        if self._server_thread and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)
        logger.info("Worker TCP server stopped")
