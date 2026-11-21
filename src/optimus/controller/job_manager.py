"""
Optimus Job Manager.
Orchestrates training lifecycle, failure detection, automatic DDP group restart,
checkpoint restoration, and final model artifact preservation.
"""

from dataclasses import asdict, dataclass
import json
import logging
import os
import shutil
import time
from typing import Any, Dict, List, Optional
import torch

from optimus.controller.checkpoint import CheckpointManager
from optimus.controller.communication import WorkerClientManager
from optimus.controller.launcher import DDPLauncher
from optimus.controller.probing import ProbeManager
from optimus.controller.scheduler import ScheduleResult, Scheduler
from optimus.dataset import DatasetManifest
from optimus.protocol.generated.messages_pb2 import (
    JobState,
    JobStatusResponse,
    WorkerState,
)

logger = logging.getLogger("optimus.controller.job_manager")

# Consecutive polls in which a rank could not confirm it is still running/completed
# (query error, JOB_UNKNOWN, JOB_STOPPED) before the attempt is treated as failed.
MAX_STATUS_MISSES = 5

# Worker errors that will fail identically on every attempt; restarting only delays
# the error. Matched against the worker's error_message (which carries the traceback).
NON_RETRYABLE_ERRORS = (
    "DatasetUnavailableError",
    "FactoryResolutionError",
    "ModuleNotFoundError",
    "ImportError",
    "SyntaxError",
)


class NonRetryableJobError(RuntimeError):
    """A worker failed in a way that a restart cannot fix (bad code path, missing data)."""


@dataclass
class JobRecord:
    job_id: str
    dataset_id: str
    dataset_version: str
    world_size: int
    status: JobState
    epoch: int = 0
    global_step: int = 0
    loss: float = 0.0
    restart_count: int = 0
    max_restarts: int = 3
    selected_nodes: List[Dict[str, Any]] = None
    last_checkpoint_path: str = ""
    error_message: str = ""
    start_time: float = 0.0
    completion_time: float = 0.0


class JobManager:
    """Oversees job scheduling, execution monitoring, and fail-stop recovery."""

    def __init__(
        self,
        scheduler: Scheduler,
        launcher: DDPLauncher,
        probe_manager: ProbeManager,
        client_manager: WorkerClientManager,
        checkpoint_base_dir: str = "/srv/checkpoints",
        runs_dir: str = "./runs",
    ):
        self.scheduler = scheduler
        self.launcher = launcher
        self.probe_manager = probe_manager
        self.client_manager = client_manager
        self.checkpoint_base_dir = checkpoint_base_dir
        self.runs_dir = os.path.abspath(runs_dir)
        os.makedirs(self.runs_dir, exist_ok=True)

        self._active_jobs: Dict[str, JobRecord] = {}

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return self._active_jobs.get(job_id)

    def list_jobs(self) -> List[JobRecord]:
        return list(self._active_jobs.values())

    def run_training_job(
        self,
        job_id: str,
        manifest: DatasetManifest,
        world_size: int = 2,
        training_args: Optional[Dict[str, Any]] = None,
        max_restarts: int = 3,
        poll_interval: float = 2.0,
    ) -> JobRecord:
        """
        Executes a distributed training run to completion, handling automatic fail-stop recovery.
        """
        training_args = training_args or {}
        checkpoint_mgr = CheckpointManager(
            base_dir=self.checkpoint_base_dir, job_id=job_id
        )

        record = JobRecord(
            job_id=job_id,
            dataset_id=manifest.dataset_id,
            dataset_version=manifest.version,
            world_size=world_size,
            status=JobState.JOB_PENDING,
            max_restarts=max_restarts,
            start_time=time.time(),
        )
        self._active_jobs[job_id] = record

        while record.restart_count <= max_restarts:
            try:
                # 1. Schedule workers
                logger.info(
                    f"Scheduling {world_size} workers for job '{job_id}' (Attempt {record.restart_count + 1}/{max_restarts + 1})..."
                )
                sched_result = self.scheduler.schedule(
                    job_id=job_id,
                    manifest=manifest,
                    world_size=world_size,
                )
                record.selected_nodes = [
                    {
                        "hostname": n.hostname,
                        "port": n.port,
                        "rank": n.rank,
                        "cache_locality": n.cache_locality,
                    }
                    for n in sched_result.selected_nodes
                ]

                # 2. Check for latest committed checkpoint to resume from
                latest_ckpt = checkpoint_mgr.get_latest_checkpoint_dir()
                resume_flag = bool(latest_ckpt)
                if resume_flag:
                    logger.info(f"Resuming job '{job_id}' from checkpoint: {latest_ckpt}")

                # 3. Launch DDP on selected workers
                self.launcher.launch(
                    schedule_result=sched_result,
                    master_addr=sched_result.selected_nodes[0].hostname,
                    master_port=29500 + (record.restart_count * 10),
                    checkpoint_dir=self.checkpoint_base_dir,
                    training_args_json=json.dumps(training_args),
                    resume_from_checkpoint=resume_flag,
                    resume_checkpoint_path=latest_ckpt or "",
                )
                record.status = JobState.JOB_RUNNING

                # 4. Monitor loop until complete or failed
                job_finished = self._monitor_job_execution(
                    record=record,
                    sched_result=sched_result,
                    checkpoint_mgr=checkpoint_mgr,
                    poll_interval=poll_interval,
                )

                if job_finished:
                    self._finalize_completed_job(record, checkpoint_mgr, training_args)
                    return record

            except NonRetryableJobError as e:
                logger.error(f"Job '{job_id}' failed with a non-retryable error; not restarting")
                record.error_message = str(e)
                self._stop_ddp_group(record)
                record.status = JobState.JOB_FAILED
                record.completion_time = time.time()
                return record

            except Exception as e:
                logger.error(f"Error during job execution: {e}")
                record.error_message = str(e)

            # Failure occurred: stop all remaining ranks in the group
            logger.warning(
                f"Failure detected for job '{job_id}'. Stopping entire DDP group..."
            )
            self._stop_ddp_group(record)

            record.restart_count += 1
            if record.restart_count > max_restarts:
                logger.error(
                    f"Job '{job_id}' exceeded maximum restart threshold ({max_restarts}). Marking FAILED."
                )
                record.status = JobState.JOB_FAILED
                record.completion_time = time.time()
                break

            logger.info(
                f"Waiting 3 seconds before restart attempt {record.restart_count}/{max_restarts}..."
            )
            time.sleep(3.0)

        return record

    def _monitor_job_execution(
        self,
        record: JobRecord,
        sched_result: ScheduleResult,
        checkpoint_mgr: CheckpointManager,
        poll_interval: float,
    ) -> bool:
        """
        Polls worker job statuses and probe health.
        Returns True if job completes successfully, False if worker failure requires restart.
        """
        selected = sched_result.selected_nodes
        misses = {node.hostname: 0 for node in selected}
        while True:
            time.sleep(poll_interval)

            # Check health probes
            self.probe_manager.probe_all()
            states = {s.hostname: s for s in self.probe_manager.get_all_states()}

            all_completed = True
            for node in selected:
                node_state = states.get(node.hostname)
                if node_state and node_state.state == WorkerState.UNAVAILABLE:
                    logger.error(f"Worker {node.hostname} failed consecutive probes (UNAVAILABLE)")
                    return False  # Trigger DDP group restart

                # Query job status over the REST API
                client = self.client_manager.get_client(node.hostname, node.port)
                try:
                    status_resp: JobStatusResponse = client.get_job_status(
                        record.job_id, timeout=2.0
                    )
                    if status_resp.status == JobState.JOB_FAILED:
                        msg = (
                            f"Rank {node.rank} on {node.hostname} failed: "
                            f"{status_resp.error_message}"
                        )
                        logger.error(msg)
                        record.error_message = msg
                        if any(e in status_resp.error_message for e in NON_RETRYABLE_ERRORS):
                            raise NonRetryableJobError(msg)
                        return False

                    if status_resp.status != JobState.JOB_COMPLETED:
                        all_completed = False
                    if status_resp.status in (JobState.JOB_RUNNING, JobState.JOB_COMPLETED):
                        misses[node.hostname] = 0
                    else:  # JOB_UNKNOWN (optimusd restarted), JOB_STOPPED, ...
                        misses[node.hostname] += 1
                        logger.warning(
                            f"Worker {node.hostname} reports job state "
                            f"{JobState.Name(status_resp.status)}: {status_resp.error_message}"
                        )

                    if node.rank == 0:
                        record.epoch = status_resp.epoch
                        record.global_step = status_resp.global_step
                        record.loss = status_resp.loss

                except NonRetryableJobError:
                    raise
                except Exception as ex:
                    # A failed query proves nothing about completion.
                    all_completed = False
                    misses[node.hostname] += 1
                    logger.warning(f"Failed to query job status on {node.hostname}: {ex}")

                if misses[node.hostname] >= MAX_STATUS_MISSES:
                    record.error_message = (
                        f"Lost track of rank {node.rank} on {node.hostname} "
                        f"({MAX_STATUS_MISSES} consecutive status checks failed)"
                    )
                    logger.error(record.error_message)
                    return False

            # Check if all ranks completed
            if all_completed:
                logger.info(f"All {len(selected)} ranks completed successfully")
                return True

    def _stop_ddp_group(self, record: JobRecord) -> None:
        """Calls POST /v1/jobs/{job_id}/stop on all assigned nodes to cleanly terminate lingering processes."""
        if not record.selected_nodes:
            return
        for node_info in record.selected_nodes:
            try:
                client = self.client_manager.get_client(
                    node_info["hostname"], node_info.get("port", 45870)
                )
                client.stop_job(
                    record.job_id, reason="Failure recovery restart", graceful=False, timeout=2.0
                )
            except Exception:
                pass

    def _finalize_completed_job(
        self,
        record: JobRecord,
        checkpoint_mgr: CheckpointManager,
        training_args: Dict[str, Any],
    ) -> None:
        """Copies final model weights and metrics into the controller run folder."""
        record.status = JobState.JOB_COMPLETED
        record.completion_time = time.time()

        job_run_dir = os.path.join(self.runs_dir, record.job_id)
        os.makedirs(job_run_dir, exist_ok=True)

        latest_ckpt = checkpoint_mgr.get_latest_checkpoint_dir()
        if latest_ckpt and os.path.isfile(os.path.join(latest_ckpt, "model.pt")):
            final_model_path = os.path.join(job_run_dir, "final_model.pt")
            shutil.copy2(os.path.join(latest_ckpt, "model.pt"), final_model_path)
            logger.info(f"Saved final model weights to '{final_model_path}'")

        # Save metrics
        metrics = {
            "job_id": record.job_id,
            "status": "COMPLETED",
            "final_epoch": record.epoch,
            "final_step": record.global_step,
            "final_loss": record.loss,
            "restarts": record.restart_count,
            "duration_seconds": record.completion_time - record.start_time,
        }
        with open(os.path.join(job_run_dir, "training_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

        # Save config
        with open(os.path.join(job_run_dir, "config.json"), "w") as f:
            json.dump(training_args, f, indent=2)

        logger.info(f"Job '{record.job_id}' finalized successfully in '{job_run_dir}'")
