"""
Optimus Controller Web Dashboard.
FastAPI web application providing cluster observability and unified scheduling.
Hosted on the controller node only.
"""

from contextlib import asynccontextmanager
import logging
import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from optimus.cluster import OptimusConfig
from optimus.protocol.generated.messages_pb2 import JobState, WorkerState
from optimus.sdk import Optimus

logger = logging.getLogger("optimus.dashboard")

templates_dir = os.path.join(os.path.dirname(__file__), "templates")
static_dir = os.path.join(os.path.dirname(__file__), "static")

# Shared Optimus cluster instance
cluster_instance: Optional[Optimus] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cluster_instance
    config_path = os.environ.get("OPTIMUS_CONFIG", "config.yaml")
    if cluster_instance is None and os.path.isfile(config_path):
        try:
            cluster_instance = Optimus(config=config_path)
            cluster_instance.probe_manager.start_probing_loop()
            logger.info("Optimus dashboard initialized cluster background manager")
        except Exception as e:
            logger.error(f"Failed to initialize Optimus cluster: {e}")
    yield
    if cluster_instance is not None:
        cluster_instance.close()


app = FastAPI(title="Optimus Cluster Dashboard", lifespan=lifespan)

# Mount static and templates
os.makedirs(templates_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=templates_dir)


class ScheduleRequest(BaseModel):
    dataset_id: str
    dataset_version: str = "v1"
    workers: int = 2
    job_id: Optional[str] = None


def get_cluster() -> Optimus:
    global cluster_instance
    if cluster_instance is None:
        config_path = os.environ.get("OPTIMUS_CONFIG", "config.yaml")
        cluster_instance = Optimus(config=config_path)
    return cluster_instance


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/api/overview")
async def get_overview():
    cluster = get_cluster()
    states = cluster.probe_manager.get_all_states()
    avail = sum(1 for s in states if s.state == WorkerState.AVAILABLE)
    busy = sum(1 for s in states if s.state == WorkerState.BUSY)
    unavail = sum(1 for s in states if s.state == WorkerState.UNAVAILABLE)

    jobs = cluster.job_manager.list_jobs()
    active_jobs = sum(1 for j in jobs if j.status == JobState.JOB_RUNNING)

    total_cached_bytes = 0
    for s in states:
        if s.state in (WorkerState.AVAILABLE, WorkerState.BUSY):
            try:
                summary = cluster.cache_coordinator.get_summary(s.hostname, s.port, "", "")
                total_cached_bytes += summary.total_cached_bytes
            except Exception:
                pass

    return {
        "controller_hostname": cluster.config.controller.hostname,
        "total_workers": len(states),
        "available_workers": avail,
        "busy_workers": busy,
        "unavailable_workers": unavail,
        "active_jobs": active_jobs,
        "total_cached_bytes": total_cached_bytes,
    }


@app.get("/api/workers")
async def get_workers():
    cluster = get_cluster()
    states = cluster.probe_manager.get_all_states()
    results = []
    for s in states:
        results.append({
            "worker_id": s.worker_id,
            "hostname": s.hostname,
            "port": s.port,
            "status": WorkerState.Name(s.state),
            "cpu_percent": s.cpu_percent,
            "ram_total_mb": s.ram_total_mb,
            "ram_available_mb": s.ram_available_mb,
            "cache_disk_available_gb": s.cache_disk_available_gb,
            "temperature_c": s.temperature_c if s.temperature_available else None,
            "active_jobs": s.active_jobs,
            "uptime_seconds": s.uptime_seconds,
            "consecutive_failures": s.consecutive_failures,
        })
    return results


@app.post("/api/probe")
async def trigger_probe():
    cluster = get_cluster()
    cluster.probe_manager.probe_all_resources()
    return {"status": "ok", "message": "Probes completed"}


@app.get("/api/cache")
async def get_cache_info(dataset_id: str = "animals", version: str = "v1"):
    cluster = get_cluster()
    states = cluster.probe_manager.get_all_states()
    results = []
    for s in states:
        try:
            summary = cluster.cache_coordinator.get_summary(s.hostname, s.port, dataset_id, version)
            results.append({
                "hostname": s.hostname,
                "dataset_id": dataset_id,
                "version": version,
                "cached_files": summary.cached_files_count,
                "cached_bytes": summary.total_cached_bytes,
                "cache_hits": summary.cache_hits,
                "cache_misses": summary.cache_misses,
                "downloaded_bytes": summary.downloaded_bytes,
                "revision": summary.registry_revision,
            })
        except Exception as e:
            results.append({
                "hostname": s.hostname,
                "dataset_id": dataset_id,
                "version": version,
                "error": str(e),
            })
    return results


@app.post("/api/schedule")
async def schedule_workers(req: ScheduleRequest):
    cluster = get_cluster()
    try:
        result = cluster.schedule(
            dataset_id=req.dataset_id,
            dataset_version=req.dataset_version,
            workers=req.workers,
            job_id=req.job_id,
        )
        return result.to_dict()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/jobs")
async def get_jobs():
    cluster = get_cluster()
    jobs = cluster.job_manager.list_jobs()
    return [
        {
            "job_id": j.job_id,
            "dataset_id": j.dataset_id,
            "dataset_version": j.dataset_version,
            "world_size": j.world_size,
            "status": JobState.Name(j.status),
            "epoch": j.epoch,
            "global_step": j.global_step,
            "loss": round(j.loss, 4),
            "restarts": j.restart_count,
            "selected_nodes": j.selected_nodes,
        }
        for j in jobs
    ]


@app.post("/api/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    cluster = get_cluster()
    job = cluster.job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    cluster.job_manager._stop_ddp_group(job)
    job.status = JobState.JOB_STOPPED
    return {"status": "ok", "message": f"Job {job_id} stopped"}
