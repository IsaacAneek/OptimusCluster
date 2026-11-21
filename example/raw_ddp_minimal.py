"""
A minimal, standalone raw PyTorch DDP + DDS demonstration.
Can be executed directly on any machine with PyTorch installed:
    python example/raw_ddp_minimal.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


# ---------------------------------------------------------
# 1. Toy Dataset: 10 items (inputs 0..9, targets 0..9 * 2)
# ---------------------------------------------------------
class ToyDataset(Dataset):
    def __init__(self):
        self.data = torch.arange(10, dtype=torch.float32).unsqueeze(1) # shape: [10, 1]
        self.targets = self.data * 2.0                                # y = 2x

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], idx


# ---------------------------------------------------------
# 2. Worker Process Function (Executed by each Rank)
# ---------------------------------------------------------
def run_worker(rank: int, world_size: int):
    # (A) Configure Rendezvous Environment
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    # (B) Rendezvous & Process Group Initialization
    # Gloo TCPStore runs on Rank 0 at 127.0.0.1:29500.
    # Every rank connects, reports its socket, and Gloo connects full mesh.
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size
    )

    print(f"[Rank {rank}] Initialized Gloo process group (world_size={world_size})")

    # (C) Dataset & Distributed Data Sampler (DDS)
    dataset = ToyDataset()
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False  # Disabled shuffle so you can clearly see index partitioning!
    )
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)

    # (D) Model & Distributed Data Parallel (DDP)
    # y = w * x (single scalar weight)
    model = nn.Linear(1, 1, bias=False)
    # Force initial weight so both start identical
    with torch.no_grad():
        model.weight.fill_(1.0)

    # DDP wrapper registers autograd backward hooks on model parameters
    ddp_model = DDP(model)
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # (E) Training Loop for 1 Epoch
    sampler.set_epoch(0)

    print(f"\n--- [Rank {rank}] Starting Epoch 0 ---")
    for batch_idx, (inputs, targets, sample_indices) in enumerate(dataloader):
        optimizer.zero_grad()

        # Print the exact sample indices this rank is processing
        print(f"[Rank {rank}] Batch {batch_idx}: processing sample indices {sample_indices.tolist()}")

        outputs = ddp_model(inputs)
        loss = criterion(outputs, targets)

        # Autograd backward + Automatic Gloo AllReduce
        # Under the hood:
        # 1. Backprop calculates local gradient dL/dw.
        # 2. DDP hook intercepts gradient bucket and calls Gloo AllReduce (SUM/AVG).
        # 3. model.weight.grad now contains the exact average of both ranks!
        loss.backward()

        print(f"[Rank {rank}] Batch {batch_idx}: local loss = {loss.item():.4f}, "
              f"synchronized weight.grad = {ddp_model.module.weight.grad.item():.4f}")

        optimizer.step()

    # (F) Verify Both Ranks Have Exactly Identical Model Weights
    final_weight = ddp_model.module.weight.item()
    print(f"\n[Rank {rank}] Final weight: {final_weight:.6f}")

    # (G) Clean up
    dist.destroy_process_group()


# ---------------------------------------------------------
# 3. Main Entry: Spawns 2 processes
# ---------------------------------------------------------
if __name__ == "__main__":
    world_size = 2
    print(f"Spawning {world_size} distributed worker processes...")
    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)
    print("\nDistributed training finished successfully!")
