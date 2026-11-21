"""
Optimus DDP Launcher.
Consumes selected_nodes.json and dispatches training jobs to assigned worker ranks over TCP.
"""

import json
import logging
import os
import socket
from typing import Dict, List, Optional

from optimus.controller.communication import WorkerClientManager
from optimus.controller.scheduler import ScheduleResult, SelectedNode
from optimus.protocol.generated.messages_pb2 import JobResponse, JobSubmit

logger = logging.getLogger("optimus.controller.launcher")


class DDPLauncher:
    """Dispatches PyTorch DDP training tasks to selected worker nodes."""

    def __init__(self, client_manager: WorkerClientManager):
        self.client_manager = client_manager

    def launch(
        self,
        schedule_result: ScheduleResult,
        master_addr: Optional[str] = None,
        master_port: int = 29500,
        checkpoint_dir: str = "/srv/checkpoints",
        training_entrypoint: str = "optimus.training.worker_entry",
        training_args_json: str = "{}",
        required_ram_mb: int = 500,
        resume_from_checkpoint: bool = False,
        resume_checkpoint_path: str = "",
    ) -> List[JobResponse]:
        """
        Reads the assigned worker ranks and submits JobSubmit requests over TCP.
        Rendezvous endpoint defaults to the Rank 0 node's hostname/IP.
        """
        selected_nodes = schedule_result.selected_nodes
        if not selected_nodes:
            raise ValueError("No workers selected for DDP launch")

        # By default, Rank 0 serves as the rendezvous master endpoint
        rank_0_node = next((n for n in selected_nodes if n.rank == 0), selected_nodes[0])
        rendezvous_addr = master_addr or rank_0_node.hostname
        try:
            rendezvous_addr = socket.gethostbyname(rendezvous_addr)
        except Exception as e:
            logger.warning(f"Could not resolve rendezvous master_addr '{rendezvous_addr}': {e}")
        world_size = len(selected_nodes)

        logger.info(
            f"Launching DDP job '{schedule_result.job_id}' across {world_size} nodes. "
            f"Rendezvous master: {rendezvous_addr}:{master_port}"
        )

        responses: List[JobResponse] = []

        for node in selected_nodes:
            client = self.client_manager.get_client(node.hostname, node.port)
            req = JobSubmit(
                job_id=schedule_result.job_id,
                dataset_id=schedule_result.dataset_id,
                dataset_version=schedule_result.dataset_version,
                rank=node.rank,
                world_size=world_size,
                master_addr=rendezvous_addr,
                master_port=master_port,
                checkpoint_dir=checkpoint_dir,
                training_entrypoint=training_entrypoint,
                training_args_json=training_args_json,
                required_ram_mb=required_ram_mb,
                resume_from_checkpoint=resume_from_checkpoint,
                resume_checkpoint_path=resume_checkpoint_path,
            )

            resp: JobResponse = client.submit_job(req)
            if not resp.accepted:
                logger.error(
                    f"Worker {node.hostname}:{node.port} rejected job '{schedule_result.job_id}': "
                    f"{resp.rejection_reason}"
                )
                raise RuntimeError(
                    f"Job submission rejected by {node.hostname}: {resp.rejection_reason}"
                )

            logger.info(
                f"Worker {node.hostname}:{node.port} accepted job '{schedule_result.job_id}' as Rank {node.rank}"
            )
            responses.append(resp)

        return responses

    @classmethod
    def load_selected_nodes_file(cls, filepath: str) -> ScheduleResult:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        nodes = [
            SelectedNode(
                hostname=n["hostname"],
                port=n.get("port", 45870),
                worker_id=n.get("worker_id", f"worker-{n['hostname']}"),
                rank=n["rank"],
                cached_bytes=n.get("cached_bytes", 0),
                required_bytes=n.get("required_bytes", 0),
                cache_locality=n.get("cache_locality", 0.0),
                score=n.get("score", 0.0),
            )
            for n in data["selected_nodes"]
        ]
        return ScheduleResult(
            job_id=data["job_id"],
            dataset_id=data["dataset_id"],
            dataset_version=data["dataset_version"],
            world_size=data["world_size"],
            output_file=filepath,
            selected_nodes=nodes,
        )
