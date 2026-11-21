"""
optimusd worker HTTP server.

Serves the Optimus control-plane REST API (see ``optimus.protocol.rest``) using
only the Python standard library (``http.server``). The controller issues all
requests; workers NEVER push heartbeats, they only answer when queried.
"""

import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
import re
import shutil
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple
from urllib.parse import parse_qs, unquote, urlsplit

from optimus.cache.manager import CacheManager
from optimus.protocol import rest
from optimus.protocol.messages import (
    BaseMessage,
    CheckpointEntry,
    CheckpointListResponse,
    ErrorResponse,
    JobState,
    JobStop,
    JobSubmit,
    ShutdownRequest,
    StatusResponse,
    WorkerState,
)
from optimus.protocol.rest import CURRENT_PROTOCOL_VERSION, DEFAULT_MAX_MESSAGE_SIZE
from optimus.worker.executor import TrainingExecutor
from optimus.worker.resource_info import ResourceCollector

logger = logging.getLogger("optimusd.server")


class ApiError(Exception):
    """Raised by a route handler to return an ErrorResponse with an HTTP status."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = int(status)
        self.code = code
        self.message = message


class Request:
    """Parsed inbound request handed to route handlers."""

    def __init__(
        self,
        method: str,
        path: str,
        path_params: Dict[str, str],
        query: Dict[str, str],
        body: bytes,
        sender: str,
    ):
        self.method = method
        self.path = path
        self.path_params = path_params
        self.query = query
        self.body = body
        self.sender = sender

    def json(self, cls):
        try:
            return rest.decode(self.body, cls)
        except rest.ProtocolError as e:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_REQUEST", str(e))
        except TypeError as e:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_REQUEST", f"Invalid body fields: {e}")

    def q_str(self, name: str, default: str = "") -> str:
        return self.query.get(name, default)

    def q_int(self, name: str, default: int = 0) -> int:
        raw = self.query.get(name)
        if raw is None or raw == "":
            return default
        try:
            return int(raw)
        except ValueError:
            raise ApiError(
                HTTPStatus.BAD_REQUEST, "BAD_REQUEST", f"Query parameter '{name}' must be an integer"
            )


Handler = Callable[[Request], Tuple[int, BaseMessage]]


class _Route:
    def __init__(self, method: str, pattern: str, handler: Handler):
        self.method = method
        # "/v1/jobs/{job_id}" -> named regex group
        regex = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]+)", pattern)
        self.regex: Pattern[str] = re.compile(f"^{regex}$")
        self.handler = handler


class _OptimusHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler_cls, worker: "WorkerServer"):
        self.worker = worker
        super().__init__(addr, handler_cls)


class _RequestHandler(BaseHTTPRequestHandler):
    server_version = "optimusd"
    protocol_version = "HTTP/1.1"

    # Route every verb through one dispatcher
    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def log_message(self, fmt, *args):  # route http.server logs to our logger
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, message: Any) -> None:
        body = rest.encode(message)
        self.send_response(status)
        self.send_header("Content-Type", rest.JSON_CONTENT_TYPE)
        self.send_header("Content-Length", str(len(body)))
        self.send_header(rest.PROTOCOL_VERSION_HEADER, str(CURRENT_PROTOCOL_VERSION))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, code: str, message: str) -> None:
        self._send(status, ErrorResponse(error_code=code, error_message=message))

    def _dispatch(self, method: str) -> None:
        worker: WorkerServer = self.server.worker
        try:
            self._handle(worker, method)
        except ApiError as e:
            self._send_error(e.status, e.code, e.message)
        except Exception as ex:  # pragma: no cover - defensive
            logger.exception(f"Error handling {method} {self.path}: {ex}")
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "INTERNAL_ERROR", str(ex))

    def _handle(self, worker: "WorkerServer", method: str) -> None:
        # Body (always read it so keep-alive connections stay in sync)
        length_hdr = self.headers.get("Content-Length", "0") or "0"
        try:
            length = int(length_hdr)
        except ValueError:
            self.close_connection = True
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_REQUEST", "Invalid Content-Length")
        if length > worker.max_message_size:
            self.close_connection = True
            raise ApiError(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "MESSAGE_TOO_LARGE",
                f"Body of {length} bytes exceeds maximum {worker.max_message_size} bytes",
            )
        body = self.rfile.read(length) if length > 0 else b""

        # Protocol version (optional header; reject explicit mismatches)
        version = self.headers.get(rest.PROTOCOL_VERSION_HEADER)
        if version is not None and version.strip() != str(CURRENT_PROTOCOL_VERSION):
            raise ApiError(
                HTTPStatus.BAD_REQUEST,
                "VERSION_MISMATCH",
                f"Supported protocol version is {CURRENT_PROTOCOL_VERSION}, received {version}",
            )

        # Authentication
        if worker.cluster_secret:
            auth = self.headers.get(rest.AUTH_HEADER, "")
            scheme, _, token = auth.partition(" ")
            if scheme != rest.AUTH_SCHEME or not hmac.compare_digest(
                token.strip().encode(), worker.cluster_secret.encode()
            ):
                logger.warning(f"Unauthorized request from {self.client_address[0]}")
                raise ApiError(
                    HTTPStatus.UNAUTHORIZED, "UNAUTHORIZED", "Invalid cluster authentication token"
                )

        # Routing
        parts = urlsplit(self.path)
        path = parts.path.rstrip("/") or "/"
        query = {k: v[-1] for k, v in parse_qs(parts.query, keep_blank_values=True).items()}

        path_matched = False
        for route in worker.routes:
            m = route.regex.match(path)
            if not m:
                continue
            path_matched = True
            if route.method != method:
                continue
            req = Request(
                method=method,
                path=path,
                path_params={k: unquote(v) for k, v in m.groupdict().items()},
                query=query,
                body=body,
                sender=self.headers.get(rest.SENDER_HEADER, ""),
            )
            status, message = route.handler(req)
            self._send(status, message)
            return

        if path_matched:
            raise ApiError(
                HTTPStatus.METHOD_NOT_ALLOWED, "METHOD_NOT_ALLOWED", f"{method} not allowed on {path}"
            )
        raise ApiError(HTTPStatus.NOT_FOUND, "NOT_FOUND", f"No route for {method} {path}")


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

        self.routes: List[_Route] = []
        self._register_default_routes()

        # Optional hook called after POST /v1/shutdown has stopped the server
        # (optimusd uses it to exit the process).
        self.on_shutdown: Optional[Callable[[], None]] = None

        self._httpd: Optional[_OptimusHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ routes

    def register_route(self, method: str, pattern: str, handler: Handler) -> None:
        """Register an extra REST route, e.g. ("GET", "/v1/foo/{id}", fn)."""
        self.routes.append(_Route(method.upper(), pattern, handler))

    def _register_default_routes(self) -> None:
        self.register_route("GET", rest.STATUS_PATH, self._handle_status)
        self.register_route("GET", rest.RESOURCES_PATH, self._handle_resources)
        self.register_route("GET", rest.CACHE_SUMMARY_PATH, self._handle_cache_summary)
        self.register_route("GET", rest.CACHE_INVENTORY_PATH, self._handle_cache_inventory)
        self.register_route("GET", rest.CHECKPOINTS_PATH, self._handle_checkpoints)
        self.register_route("POST", rest.JOBS_PATH, self._handle_job_submit)
        self.register_route("GET", rest.JOBS_PATH + "/{job_id}", self._handle_job_status)
        self.register_route("POST", rest.JOBS_PATH + "/{job_id}/stop", self._handle_job_stop)
        self.register_route("POST", rest.SHUTDOWN_PATH, self._handle_shutdown)

    def _status_response(self, status: Optional[int] = None) -> StatusResponse:
        if status is None:
            status = self.state if self.active_jobs == 0 else WorkerState.BUSY
        return StatusResponse(
            worker_id=self.worker_id,
            hostname=self.hostname,
            protocol_version=CURRENT_PROTOCOL_VERSION,
            status=status,
            active_job_count=self.active_jobs,
            uptime_seconds=int(time.time() - self.start_time),
        )

    def _handle_status(self, req: Request):
        return HTTPStatus.OK, self._status_response()

    def _handle_resources(self, req: Request):
        resp = self.resource_collector.collect(
            worker_id=self.worker_id,
            hostname=self.hostname,
            current_state=self.state,
            active_jobs=self.active_jobs,
        )
        return HTTPStatus.OK, resp

    def _handle_cache_summary(self, req: Request):
        resp = self.cache_manager.build_summary_response(
            worker_id=self.worker_id,
            dataset_id=req.q_str("dataset_id"),
            dataset_version=req.q_str("dataset_version"),
        )
        return HTTPStatus.OK, resp

    def _handle_cache_inventory(self, req: Request):
        resp = self.cache_manager.build_inventory_response(
            worker_id=self.worker_id,
            dataset_id=req.q_str("dataset_id"),
            dataset_version=req.q_str("dataset_version"),
            known_revision=req.q_int("known_revision", 0),
            page_number=req.q_int("page", 1) or 1,
            page_size=req.q_int("page_size", 500) or 500,
        )
        return HTTPStatus.OK, resp

    def _handle_checkpoints(self, req: Request):
        target_dir = req.q_str("checkpoint_dir") or "/srv/checkpoints"
        return HTTPStatus.OK, self._scan_checkpoints(target_dir, req.q_str("job_id"))

    def _handle_job_submit(self, req: Request):
        submit: JobSubmit = req.json(JobSubmit)
        if not submit.job_id:
            raise ApiError(HTTPStatus.BAD_REQUEST, "BAD_REQUEST", "job_id is required")
        resp = self.executor.submit_job(submit)
        if resp.accepted:
            self.active_jobs = 1
            return HTTPStatus.CREATED, resp
        return HTTPStatus.CONFLICT, resp

    def _handle_job_status(self, req: Request):
        resp = self.executor.get_status(req.path_params["job_id"], self.worker_id)
        if resp.status in (JobState.JOB_COMPLETED, JobState.JOB_FAILED, JobState.JOB_STOPPED):
            self.active_jobs = 0
        return HTTPStatus.OK, resp

    def _handle_job_stop(self, req: Request):
        stop: JobStop = req.json(JobStop)
        self.executor.stop_job(req.path_params["job_id"], stop.reason)
        self.active_jobs = 0
        return HTTPStatus.OK, self._status_response(WorkerState.AVAILABLE)

    def _handle_shutdown(self, req: Request):
        shutdown: ShutdownRequest = req.json(ShutdownRequest)
        logger.info(f"Shutdown requested by {req.sender or 'unknown'}: {shutdown.reason}")
        # Stop after the response has been written
        threading.Timer(0.2, self._shutdown_from_api).start()
        return HTTPStatus.ACCEPTED, self._status_response(WorkerState.UNAVAILABLE)

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

    # --------------------------------------------------------------- lifecycle

    def start(self) -> None:
        """Starts the HTTP server in a background thread."""
        self._httpd = _OptimusHTTPServer((self.host, self.port), _RequestHandler, self)
        self._server_thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.5},
            name=f"optimusd-http-{self.port}",
            daemon=True,
        )
        self._server_thread.start()
        logger.info(
            f"optimusd REST API listening on http://{self.host}:{self.port}{rest.API_PREFIX} "
            f"(ID: {self.worker_id})"
        )

    def _shutdown_from_api(self) -> None:
        self.stop()
        if self.on_shutdown:
            self.on_shutdown()

    def stop(self) -> None:
        """Stops the HTTP server."""
        httpd = self._httpd
        if httpd is None:
            return
        self._httpd = None
        if threading.current_thread() is not self._server_thread:
            httpd.shutdown()
        httpd.server_close()
        if (
            self._server_thread
            and self._server_thread.is_alive()
            and threading.current_thread() is not self._server_thread
        ):
            self._server_thread.join(timeout=2.0)
        logger.info("optimusd REST API stopped")
