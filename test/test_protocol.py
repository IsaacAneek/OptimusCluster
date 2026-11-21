"""
Tests for the optimusd REST control-plane API: routing, JSON bodies, auth,
protocol-version checks, size limits and error responses.
"""

import http.client
import json
import socket

import pytest

from optimus.controller.communication import (
    WorkerClient,
    WorkerCommunicationError,
    WorkerRequestError,
)
from optimus.protocol import rest
from optimus.protocol.messages import (
    CacheInventoryResponse,
    CachedFileEntry,
    JobState,
    JobSubmit,
    ResourceResponse,
    StatusResponse,
    ThermalZone,
    WorkerState,
)
from optimus.worker.server import WorkerServer


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@pytest.fixture
def worker(tmp_path):
    port = get_free_port()
    server = WorkerServer(
        worker_id="pi-rest-01",
        hostname="localhost",
        port=port,
        host="127.0.0.1",
        cluster_secret="s3cret",
        max_message_size=64 * 1024,
        cache_dir=str(tmp_path / "cache"),
        logs_dir=str(tmp_path / "logs"),
    )
    server.start()
    yield server
    server.stop()


def raw_request(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        return resp.status, dict(resp.getheaders()), resp.read()
    finally:
        conn.close()


def auth():
    return {"Authorization": "Bearer s3cret"}


# ------------------------------------------------------------ message codec


def test_message_json_roundtrip():
    resp = ResourceResponse(
        worker_id="w1",
        cpu_percent=12.5,
        thermals=[ThermalZone(zone_name="cpu", temperature_c=48.0)],
    )
    decoded = rest.decode(rest.encode(resp), ResourceResponse)
    assert decoded == resp
    assert isinstance(decoded.thermals[0], ThermalZone)

    inv = CacheInventoryResponse(files=[CachedFileEntry(relative_path="a.bin", file_size=3)])
    assert rest.decode(rest.encode(inv), CacheInventoryResponse).files[0].relative_path == "a.bin"


def test_decode_ignores_unknown_fields_and_rejects_bad_json():
    msg = rest.decode(b'{"worker_id": "w", "future_field": 1}', StatusResponse)
    assert msg.worker_id == "w"
    with pytest.raises(rest.ProtocolError):
        rest.decode(b"{not json", StatusResponse)
    with pytest.raises(rest.ProtocolError):
        rest.decode(b"[1, 2]", StatusResponse)


def test_job_paths_are_url_encoded():
    assert rest.job_path("job 1/x") == "/v1/jobs/job%201%2Fx"
    assert rest.job_stop_path("j1") == "/v1/jobs/j1/stop"


# ------------------------------------------------------------ server + client


def test_status_endpoint_raw_http(worker):
    status, headers, body = raw_request(worker.port, "GET", "/v1/status", headers=auth())
    assert status == 200
    assert headers["Content-Type"] == "application/json"
    assert headers[rest.PROTOCOL_VERSION_HEADER] == "1"
    data = json.loads(body)
    assert data["worker_id"] == "pi-rest-01"
    assert data["status"] == WorkerState.AVAILABLE


def test_client_typed_endpoints(worker):
    client = WorkerClient("127.0.0.1", worker.port, timeout=3, cluster_secret="s3cret")

    st = client.get_status()
    assert isinstance(st, StatusResponse) and st.worker_id == "pi-rest-01"

    res = client.get_resources()
    assert res.worker_id == "pi-rest-01" and res.ram_total_mb > 0

    summary = client.get_cache_summary("ds", "v1")
    assert summary.dataset_id == "ds" and summary.dataset_version == "v1"

    inv = client.get_cache_inventory("ds", "v1", page=1, page_size=10)
    assert inv.dataset_id == "ds" and inv.files == []

    ckpts = client.list_checkpoints(checkpoint_dir="/definitely/missing/dir")
    assert ckpts.accessible is False and "does not exist" in ckpts.error_message

    job = client.get_job_status("no-such-job")
    assert job.status == JobState.JOB_UNKNOWN

    stopped = client.stop_job("no-such-job", reason="test")
    assert stopped.status == WorkerState.AVAILABLE


def test_auth_required(worker):
    status, _, body = raw_request(worker.port, "GET", "/v1/status")
    assert status == 401
    assert json.loads(body)["error_code"] == "UNAUTHORIZED"

    bad = WorkerClient("127.0.0.1", worker.port, timeout=3, cluster_secret="wrong")
    with pytest.raises(WorkerRequestError) as ei:
        bad.get_status()
    assert ei.value.status == 401 and ei.value.error_code == "UNAUTHORIZED"


def test_version_mismatch_rejected(worker):
    headers = {**auth(), rest.PROTOCOL_VERSION_HEADER: "99"}
    status, _, body = raw_request(worker.port, "GET", "/v1/status", headers=headers)
    assert status == 400
    assert json.loads(body)["error_code"] == "VERSION_MISMATCH"


def test_unknown_route_and_wrong_method(worker):
    status, _, body = raw_request(worker.port, "GET", "/v1/nope", headers=auth())
    assert status == 404 and json.loads(body)["error_code"] == "NOT_FOUND"

    status, _, body = raw_request(worker.port, "POST", "/v1/status", body=b"{}", headers=auth())
    assert status == 405 and json.loads(body)["error_code"] == "METHOD_NOT_ALLOWED"


def test_bad_bodies_rejected(worker):
    status, _, body = raw_request(worker.port, "POST", "/v1/jobs", body=b"{oops", headers=auth())
    assert status == 400 and json.loads(body)["error_code"] == "BAD_REQUEST"

    status, _, body = raw_request(worker.port, "POST", "/v1/jobs", body=b"{}", headers=auth())
    assert status == 400 and "job_id" in json.loads(body)["error_message"]

    status, _, body = raw_request(
        worker.port, "GET", "/v1/cache/inventory?page=abc", headers=auth()
    )
    assert status == 400


def test_body_too_large(worker):
    big = b"{" + b" " * (worker.max_message_size + 10) + b"}"
    status, _, body = raw_request(worker.port, "POST", "/v1/jobs", body=big, headers=auth())
    assert status == 413
    assert json.loads(body)["error_code"] == "MESSAGE_TOO_LARGE"

    client = WorkerClient(
        "127.0.0.1", worker.port, cluster_secret="s3cret", max_message_size=50
    )
    with pytest.raises(rest.MessageTooLargeError):
        client.submit_job(JobSubmit(job_id="x" * 100))


def test_job_rejection_returns_409(worker, monkeypatch):
    from optimus.protocol.messages import JobResponse

    monkeypatch.setattr(
        worker.executor,
        "submit_job",
        lambda req: JobResponse(job_id=req.job_id, accepted=False, rejection_reason="busy"),
    )
    status, _, body = raw_request(
        worker.port, "POST", "/v1/jobs", body=rest.encode(JobSubmit(job_id="j1")), headers=auth()
    )
    assert status == 409 and json.loads(body)["accepted"] is False

    client = WorkerClient("127.0.0.1", worker.port, cluster_secret="s3cret")
    resp = client.submit_job(JobSubmit(job_id="j1"))
    assert resp.accepted is False and resp.rejection_reason == "busy"


def test_custom_route_registration(worker):
    worker.register_route(
        "GET", "/v1/echo/{name}", lambda req: (200, StatusResponse(worker_id=req.path_params["name"]))
    )
    client = WorkerClient("127.0.0.1", worker.port, cluster_secret="s3cret")
    _, data = client.request("GET", "/v1/echo/hello%20world")
    assert data["worker_id"] == "hello world"


def test_shutdown_endpoint(tmp_path):
    port = get_free_port()
    server = WorkerServer(
        worker_id="pi-bye",
        hostname="localhost",
        port=port,
        host="127.0.0.1",
        cache_dir=str(tmp_path / "cache"),
        logs_dir=str(tmp_path / "logs"),
    )
    server.start()
    client = WorkerClient("127.0.0.1", port, timeout=2)
    resp = client.shutdown(reason="test")
    assert resp.status == WorkerState.UNAVAILABLE

    import time

    deadline = time.time() + 3
    while time.time() < deadline:
        try:
            client.get_status()
        except WorkerCommunicationError:
            break
        time.sleep(0.1)
    else:
        server.stop()
        pytest.fail("Server still answering after shutdown")


def test_connection_refused_is_communication_error():
    client = WorkerClient("127.0.0.1", get_free_port(), timeout=0.5)
    with pytest.raises(WorkerCommunicationError):
        client.get_status()
