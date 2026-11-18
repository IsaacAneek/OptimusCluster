"""
Optimus PyTorch DDP Utilities.
Configures and initializes DistributedDataParallel with the Gloo CPU backend.
"""

from datetime import timedelta
import logging
import os
from typing import Optional, Tuple
import torch
import torch.distributed as dist

logger = logging.getLogger("optimus.training.ddp")


def init_distributed(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    master_addr: Optional[str] = None,
    master_port: Optional[int] = None,
    backend: str = "gloo",
    timeout_seconds: int = 120,
) -> Tuple[int, int]:
    """
    Initializes the PyTorch distributed process group over Gloo.
    Reads from parameters or standard torchrun environment variables.
    """
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    env_rank = rank if rank is not None else int(os.environ.get("RANK", 0))
    env_world_size = world_size if world_size is not None else int(os.environ.get("WORLD_SIZE", 1))
    env_addr = master_addr or os.environ.get("MASTER_ADDR", "127.0.0.1")
    env_port = str(master_port or os.environ.get("MASTER_PORT", 29500))

    os.environ["RANK"] = str(env_rank)
    os.environ["WORLD_SIZE"] = str(env_world_size)
    os.environ["MASTER_ADDR"] = env_addr
    os.environ["MASTER_PORT"] = env_port

    logger.info(
        f"Initializing PyTorch DDP process: Rank {env_rank}/{env_world_size} "
        f"(Rendezvous: {env_addr}:{env_port}, Backend: {backend})"
    )

    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{env_addr}:{env_port}",
        rank=env_rank,
        world_size=env_world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )

    return env_rank, env_world_size


def cleanup_distributed() -> None:
    """Destroys active distributed process group."""
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed")
