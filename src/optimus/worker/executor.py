"""
Optimus Worker Training Executor.
Spawns and manages training subprocesses for assigned DDP jobs.
Validates resource headroom, monitors execution status, and handles termination.
"""

from dataclasses import dataclass
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Optional, Tuple
import psutil

from optimus.protocol.generated.messages_pb2 import (
    JobResponse,
    JobState,
    JobStatusResponse,
    JobSubmit,
)

logger = logging.getLogger("optimus.worker.executor")


@dataclass
class JobExecutionRecord:
    job_id: str
    rank: int
    world_size: int
    dataset_id: str
    dataset_version: str
    status: JobState
    epoch: int = 0
    global_step: int = 0
    loss: float = 0.0
    throughput: float = 0.0
    error_message: str = ""
    start_time: float = 0.0
    process: Optional[subprocess.Popen] = None


class TrainingExecutor:
    """Manages active training process on the worker node."""

    def __init__(self, logs_dir: str = "./logs"):
        self.logs_dir = os.path.abspath(logs_dir)
        os.makedirs(self.logs_dir, exist_ok=True)
        self.current_job: Optional[JobExecutionRecord] = None
        self._lock = threading.Lock()

    def submit_job(self, req: JobSubmit) -> JobResponse:
        """
        Validates system capacity, reserves resources, and launches training subprocess.
        """
        with self._lock:
            # Check if already running a job
            if self.current_job and self.current_job.status == JobState.JOB_RUNNING:
                return JobResponse(
                    job_id=req.job_id,
                    accepted=False,
                    rejection_reason="Worker is already executing an active training job",
                )

            # Check RAM headroom
            mem = psutil.virtual_memory()
            avail_mb = int(mem.available / (1024 * 1024))
            if req.required_ram_mb > 0 and avail_mb < req.required_ram_mb:
                return JobResponse(
                    job_id=req.job_id,
                    accepted=False,
                    rejection_reason=f"Insufficient available RAM: {avail_mb} MB available, {req.required_ram_mb} MB required",
                )

            # Build environment variables for DDP
            env = os.environ.copy()
            env["RANK"] = str(req.rank)
            env["WORLD_SIZE"] = str(req.world_size)
            env["MASTER_ADDR"] = req.master_addr
            env["MASTER_PORT"] = str(req.master_port)
            env["OPTIMUS_JOB_ID"] = req.job_id
            env["OPTIMUS_DATASET_ID"] = req.dataset_id
            env["OPTIMUS_DATASET_VERSION"] = req.dataset_version
            env["OPTIMUS_CHECKPOINT_DIR"] = req.checkpoint_dir
            if req.resume_from_checkpoint and req.resume_checkpoint_path:
                env["OPTIMUS_RESUME_CHECKPOINT"] = req.resume_checkpoint_path

            # Subprocess entrypoint
            entrypoint = req.training_entrypoint or "optimus.training.worker_entry"
            cmd = [sys.executable, "-m", entrypoint]

            # Pass optional training args json if provided
            if req.training_args_json:
                cmd.extend(["--training-args", req.training_args_json])

            log_file_path = os.path.join(self.logs_dir, f"{req.job_id}_rank{req.rank}.log")
            log_f = open(log_file_path, "a", encoding="utf-8")

            try:
                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid,  # create separate process group for clean teardown
                )
            except Exception as e:
                log_f.close()
                return JobResponse(
                    job_id=req.job_id,
                    accepted=False,
                    rejection_reason=f"Failed to spawn training process: {e}",
                )

            record = JobExecutionRecord(
                job_id=req.job_id,
                rank=req.rank,
                world_size=req.world_size,
                dataset_id=req.dataset_id,
                dataset_version=req.dataset_version,
                status=JobState.JOB_RUNNING,
                start_time=time.time(),
                process=proc,
            )
            self.current_job = record

            # Launch background monitor thread for process exit
            t = threading.Thread(
                target=self._monitor_process,
                args=(record, proc, log_f),
                daemon=True,
            )
            t.start()

            logger.info(
                f"Accepted job '{req.job_id}' (Rank {req.rank}/{req.world_size}, PID: {proc.pid})"
            )
            return JobResponse(
                job_id=req.job_id,
                accepted=True,
                reserved_ram_mb=req.required_ram_mb,
            )

    def _monitor_process(
        self, record: JobExecutionRecord, proc: subprocess.Popen, log_file
    ) -> None:
        try:
            exit_code = proc.wait()
            with self._lock:
                if record.status == JobState.JOB_RUNNING:
                    if exit_code == 0:
                        record.status = JobState.JOB_COMPLETED
                        logger.info(f"Job '{record.job_id}' Rank {record.rank} COMPLETED")
                    else:
                        record.status = JobState.JOB_FAILED
                        record.error_message = f"Process exited with code {exit_code}"
                        logger.error(f"Job '{record.job_id}' Rank {record.rank} FAILED ({exit_code})")
        finally:
            try:
                log_file.close()
            except Exception:
                pass

    def get_status(self, job_id: str, worker_id: str) -> JobStatusResponse:
        """Returns the current execution status of a job."""
        with self._lock:
            if not self.current_job or self.current_job.job_id != job_id:
                return JobStatusResponse(
                    job_id=job_id,
                    worker_id=worker_id,
                    status=JobState.JOB_UNKNOWN,
                    error_message=f"Job '{job_id}' not found on this worker",
                )

            rec = self.current_job
            # Try reading status file written by worker_entry if available
            status_file = os.path.join(self.logs_dir, f"{job_id}_status.json")
            if os.path.isfile(status_file):
                try:
                    with open(status_file, "r") as f:
                        data = json.load(f)
                    rec.epoch = data.get("epoch", rec.epoch)
                    rec.global_step = data.get("global_step", rec.global_step)
                    rec.loss = data.get("loss", rec.loss)
                    rec.throughput = data.get("throughput", rec.throughput)
                except Exception:
                    pass

            return JobStatusResponse(
                job_id=job_id,
                worker_id=worker_id,
                rank=rec.rank,
                status=rec.status,
                epoch=rec.epoch,
                global_step=rec.global_step,
                loss=rec.loss,
                throughput_samples_per_sec=rec.throughput,
                error_message=rec.error_message,
            )

    def stop_job(self, job_id: str, reason: str = "") -> bool:
        """Gracefully terminates training subprocess."""
        with self._lock:
            if not self.current_job or self.current_job.job_id != job_id:
                return False

            rec = self.current_job
            proc = rec.process
            if proc and proc.poll() is None:
                logger.info(f"Stopping job '{job_id}' (PID: {proc.pid}). Reason: {reason}")
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    for _ in range(20):
                        if proc.poll() is not None:
                            break
                        time.sleep(0.1)
                    if proc.poll() is None:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception as e:
                    logger.warning(f"Error terminating process: {e}")

            rec.status = JobState.JOB_STOPPED
            rec.error_message = reason
            return True
