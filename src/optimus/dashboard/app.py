"""
Optimus Controller Web Dashboard.
FastAPI web application providing cluster observability, LAN topology discovery,
and unified scheduling.
Hosted on the controller node only.
"""

import asyncio
from contextlib import asynccontextmanager
import datetime
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from optimus.cluster import OptimusConfig
from optimus.protocol.messages import (
    CheckpointEntry,
    CheckpointListResponse,
    JobState,
    WorkerState,
)
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

    checkpoint_dir = cluster.config.training.checkpoint_directory or "/srv/checkpoints"
    total_checkpoints = 0
    if os.path.isdir(checkpoint_dir):
        try:
            for j in os.listdir(checkpoint_dir):
                jpath = os.path.join(checkpoint_dir, j)
                if os.path.isdir(jpath):
                    total_checkpoints += sum(
                        1 for c in os.listdir(jpath) if c.startswith("checkpoint-") and not c.endswith(".tmp")
                    )
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
        "total_checkpoints": total_checkpoints,
        "subnet": cluster.config.discovery.subnet or "192.168.10.0/24",
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
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, cluster.probe_manager.probe_all)
    return {"status": "ok", "message": "Probes completed"}


@app.get("/api/discovery")
async def get_discovery():
    cluster = get_cluster()
    cached = cluster.discovery_manager.get_cached_lan_hosts()
    if not cached:
        # If cache empty, run quick scan in executor
        loop = asyncio.get_event_loop()
        cached = await loop.run_in_executor(None, cluster.discovery_manager.scan_lan_topology)
    return [h.to_dict() for h in cached]


@app.post("/api/discovery/scan")
async def scan_lan():
    cluster = get_cluster()
    loop = asyncio.get_event_loop()
    hosts = await loop.run_in_executor(None, cluster.discovery_manager.scan_lan_topology)
    return [h.to_dict() for h in hosts]


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


def _scan_local_checkpoints(base_dir: str, runs_dir: str) -> List[Dict[str, Any]]:
    base_dir = os.path.abspath(base_dir)
    runs_dir = os.path.abspath(runs_dir)
    results = []
    if not os.path.isdir(base_dir):
        return results

    for job_name in os.listdir(base_dir):
        job_path = os.path.join(base_dir, job_name)
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

        saved_by_node = "Unknown"
        participating_nodes = []
        sched_file = os.path.join(runs_dir, job_name, "selected_nodes.json")
        if os.path.isfile(sched_file):
            try:
                with open(sched_file, "r") as f:
                    sdata = json.load(f)
                nodes = sdata.get("selected_nodes", [])
                for n in nodes:
                    participating_nodes.append(n.get("hostname", ""))
                    if n.get("rank") == 0:
                        saved_by_node = f"{n.get('hostname')} ({n.get('worker_id', 'Rank 0')})"
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
                            sz = os.path.getsize(fpath)
                            total_bytes += sz
                            if fname == "model.pt":
                                model_bytes = sz
                            mt = os.path.getmtime(fpath)
                            if mt > mtime:
                                mtime = mt
                except Exception:
                    pass

                epoch = meta_dict.get("epoch", 0)
                if "epoch" not in meta_dict and item.startswith("checkpoint-"):
                    try:
                        epoch = int(item.split("-")[1])
                    except Exception:
                        pass

                created_str = (
                    datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                    if mtime > 0
                    else "-"
                )

                results.append({
                    "job_id": job_name,
                    "checkpoint_name": item,
                    "epoch": epoch,
                    "global_step": meta_dict.get("global_step", 0),
                    "loss": round(float(meta_dict.get("loss", 0.0)), 4),
                    "dataset_id": meta_dict.get("dataset_id", "default"),
                    "dataset_version": meta_dict.get("dataset_version", "v1"),
                    "world_size": meta_dict.get("world_size", 1),
                    "model_size_mb": round(model_bytes / (1024 * 1024), 2),
                    "total_size_mb": round(total_bytes / (1024 * 1024), 2),
                    "is_latest": (item == latest_target),
                    "saved_by_node": saved_by_node,
                    "participating_nodes": participating_nodes,
                    "saved_files": files,
                    "mtime": mtime,
                    "created_at": created_str,
                    "path": ckpt_dir,
                    "verified_nodes": [],
                })

    results.sort(key=lambda x: (x["job_id"], x["epoch"]), reverse=True)
    return results


async def collect_checkpoints_data(cluster: Optimus) -> Dict[str, Any]:
    base_dir = cluster.config.training.checkpoint_directory or "/srv/checkpoints"
    runs_dir = cluster.config.scheduler.output_directory or "./runs"
    loop = asyncio.get_event_loop()

    # 1. Local controller scan
    controller_ckpts = await loop.run_in_executor(
        None, _scan_local_checkpoints, base_dir, runs_dir
    )

    nodes_data = []
    controller_host = cluster.config.controller.hostname or "isaac.local"

    controller_avail_gb = 0
    try:
        controller_avail_gb = int(shutil.disk_usage(base_dir).free / (1024 * 1024 * 1024))
    except Exception:
        pass

    nodes_data.append({
        "hostname": controller_host,
        "worker_id": "controller",
        "role": "Controller",
        "status": "ONLINE",
        "accessible": os.path.isdir(base_dir),
        "checkpoint_dir": base_dir,
        "disk_available_gb": controller_avail_gb,
        "checkpoints_count": len(controller_ckpts),
    })

    worker_ckpts_by_node: Dict[str, List[Any]] = {}

    for w_cfg in cluster.config.workers:
        if not w_cfg.enabled:
            continue
        hostname = w_cfg.hostname
        port = getattr(w_cfg, "port", 45870)

        node_info = {
            "hostname": hostname,
            "worker_id": f"worker-{hostname.split('.')[0]}",
            "role": "Worker",
            "status": "UNAVAILABLE",
            "accessible": False,
            "checkpoint_dir": base_dir,
            "disk_available_gb": 0,
            "checkpoints_count": 0,
        }

        try:
            client = cluster.client_manager.get_client(hostname, port)
            c_resp = await loop.run_in_executor(
                None, lambda: client.list_checkpoints(checkpoint_dir=base_dir, timeout=2.5)
            )

            if c_resp:
                node_info["status"] = "AVAILABLE" if c_resp.accessible else "ERROR"
                node_info["accessible"] = c_resp.accessible
                node_info["worker_id"] = c_resp.worker_id or node_info["worker_id"]
                node_info["disk_available_gb"] = int(
                    c_resp.disk_available_bytes / (1024 * 1024 * 1024)
                )
                node_info["checkpoints_count"] = len(c_resp.checkpoints)
                worker_ckpts_by_node[hostname] = c_resp.checkpoints
        except Exception as e:
            node_info["error_message"] = str(e)

        nodes_data.append(node_info)

    # Correlate verified_nodes for each checkpoint
    for ckpt in controller_ckpts:
        verified = [controller_host]
        for w_host, w_list in worker_ckpts_by_node.items():
            if any(c.job_id == ckpt["job_id"] and c.checkpoint_name == ckpt["checkpoint_name"] for c in w_list):
                verified.append(w_host)
        ckpt["verified_nodes"] = verified

    total_size_mb = sum(c["total_size_mb"] for c in controller_ckpts)

    summary = {
        "total_checkpoints": len(controller_ckpts),
        "total_size_mb": round(total_size_mb, 2),
        "nodes_count": len(nodes_data),
        "nodes_accessible_count": sum(1 for n in nodes_data if n["accessible"]),
        "latest_job": controller_ckpts[0]["job_id"] if controller_ckpts else "-",
        "latest_checkpoint": controller_ckpts[0]["checkpoint_name"] if controller_ckpts else "-",
    }

    return {
        "summary": summary,
        "nodes": nodes_data,
        "checkpoints": controller_ckpts,
    }


@app.get("/api/checkpoints")
async def get_checkpoints():
    cluster = get_cluster()
    return await collect_checkpoints_data(cluster)


@app.get("/api/checkpoints/{job_id}/{checkpoint_name}/metadata")
async def get_checkpoint_metadata(job_id: str, checkpoint_name: str):
    cluster = get_cluster()
    base_dir = cluster.config.training.checkpoint_directory or "/srv/checkpoints"
    meta_path = os.path.join(base_dir, job_id, checkpoint_name, "metadata.json")
    if not os.path.isfile(meta_path):
        raise HTTPException(status_code=404, detail="Checkpoint metadata not found")
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
