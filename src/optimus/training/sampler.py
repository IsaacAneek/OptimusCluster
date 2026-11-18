"""
Optimus Dataset Partitioning and Distributed Sampling.
Wraps PyTorch DistributedSampler for deterministic data partitioning across ranks.
"""

from typing import Iterator, Optional
import torch
from torch.utils.data import Dataset, DistributedSampler


class OptimusDistributedSampler(DistributedSampler):
    """
    Standard PyTorch DistributedSampler with deterministic seed handling
    and compatible step iteration across CPU workers.
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)
