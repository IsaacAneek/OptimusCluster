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
    """The index page is an Alpine.js app styled with Pico CSS and the Carbon-based theme."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html = response.text
    assert "Optimus" in html
    assert 'x-data="dashboard"' in html
    assert "/static/vendor/pico.min.css" in html
    assert "/static/vendor/alpine.min.js" in html
    assert "/static/theme.css" in html
    assert "/static/app.js" in html


def test_dashboard_static_assets_are_served_locally(client):
    """Alpine, Pico and IBM Plex are vendored so the dashboard works on an offline LAN."""
    for path, ctype in [
        ("/static/vendor/alpine.min.js", "javascript"),
        ("/static/vendor/pico.min.css", "css"),
        ("/static/theme.css", "css"),
        ("/static/app.js", "javascript"),
        ("/static/vendor/fonts/ibm-plex-sans-400.woff2", "font"),
        ("/static/vendor/fonts/ibm-plex-mono-400.woff2", "font"),
    ]:
        r = client.get(path)
        assert r.status_code == 200, path
        assert ctype in r.headers["content-type"], path
        assert len(r.content) > 1000, path
    # The old hand-written DOM script and stylesheet are gone
    assert client.get("/static/main.js").status_code == 404
    assert client.get("/static/style.css").status_code == 404


def test_dashboard_views_declared(client):
    """Navigation labels live in the Alpine component."""
    js = client.get("/static/app.js").text
    for label in ("Overview", "LAN Discovery", "Workers", "Scheduler", "Training Jobs", "Dataset Cache", "Checkpoints"):
        assert label in js


def test_dashboard_uses_only_three_theme_colors(client):
    """Theme defines exactly paper/ink/primary; no other hex colors are introduced."""
    import re

    css = client.get("/static/theme.css").text
    hexes = set(re.findall(r"#[0-9a-fA-F]{3,8}\b", css))
    assert hexes == {"#f4f4f4", "#161616", "#0f62fe"}


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
