"""
Tests for Optimus Web Dashboard.
Verifies HTML template rendering, static mounts, and REST API endpoints including LAN discovery.
"""

from fastapi.testclient import TestClient
import pytest

from optimus.dashboard.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_dashboard_index_html(client):
    """Verifies that the index page renders valid HTML and passes request context correctly."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Optimus" in response.text
    assert "Overview" in response.text
    assert "LAN Discovery" in response.text


def test_dashboard_api_overview(client):
    """Verifies the /api/overview endpoint returns valid JSON summary."""
    response = client.get("/api/overview")
    assert response.status_code == 200
    data = response.json()
    assert "available_workers" in data
    assert "active_jobs" in data
    assert "total_cached_bytes" in data


def test_dashboard_api_workers(client):
    """Verifies the /api/workers endpoint returns worker node array."""
    response = client.get("/api/workers")
    assert response.status_code == 200
    workers = response.json()
    assert isinstance(workers, list)


def test_dashboard_api_discovery(client):
    """Verifies the /api/discovery endpoint returns discovered LAN host array."""
    response = client.get("/api/discovery")
    assert response.status_code == 200
    hosts = response.json()
    assert isinstance(hosts, list)


def test_dashboard_api_jobs(client):
    """Verifies the /api/jobs endpoint returns job list."""
    response = client.get("/api/jobs")
    assert response.status_code == 200
    jobs = response.json()
    assert isinstance(jobs, list)


def test_dashboard_api_cache(client):
    """Verifies the /api/cache endpoint returns cache summary array."""
    response = client.get("/api/cache")
    assert response.status_code == 200
    cache = response.json()
    assert isinstance(cache, list)


def test_dashboard_api_checkpoints(client):
    """Verifies the /api/checkpoints endpoint returns structured node and checkpoint arrays."""
    response = client.get("/api/checkpoints")
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "nodes" in data
    assert "checkpoints" in data
    assert isinstance(data["nodes"], list)
    assert isinstance(data["checkpoints"], list)
    assert "total_checkpoints" in data["summary"]
    assert "nodes_count" in data["summary"]
