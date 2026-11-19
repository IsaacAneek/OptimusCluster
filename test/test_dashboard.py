"""
Tests for Optimus Web Dashboard.
Verifies HTML template rendering, static mounts, and REST API endpoints.
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
