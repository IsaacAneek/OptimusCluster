# Optimus: Lightweight Distributed ML Training System

Optimus is an ultra-lightweight, resilient distributed machine learning training platform engineered specifically for edge clusters composed of multiple Raspberry Pi 5 nodes coordinated by a designated Linux controller workstation (`isaac.local`).

Optimus abstracts distributed infrastructure so ML developers write ordinary PyTorch training scripts using the custom Optimus SDK. The backend transparently orchestrates worker discovery, controller-initiated health and telemetry probing, full-file FUSE caching over NFS, locality-aware scheduling, PyTorch DistributedDataParallel (DDP) over Gloo, atomic checkpointing, and fail-stop recovery.

---

## Table of Contents

1. [Architectural Principles & The 80/20 Rule](#1-architectural-principles--the-8020-rule)
2. [Cluster Topology & Hardware Model](#2-cluster-topology--hardware-model)
3. [Binary Framing & Protocol Buffers Specification](#3-binary-framing--protocol-buffers-specification)
4. [Authoritative Dataset & Full-File FUSE Caching Architecture](#4-authoritative-dataset--full-file-fuse-caching-architecture)
5. [Worker SQLite Cache Registry](#5-worker-sqlite-cache-registry)
6. [Locality-Aware Weighted Scheduler](#6-locality-aware-weighted-scheduler)
7. [PyTorch DDP & Fail-Stop Fault Tolerance](#7-pytorch-ddp--fail-stop-fault-tolerance)
8. [Web Dashboard & REST Observability](#8-web-dashboard--rest-observability)
9. [Developer-Facing SDK Manual](#9-developer-facing-sdk-manual)
10. [Installation & Deployment Guide](#10-installation--deployment-guide)
11. [Performance Evaluation & Benchmark Results](#11-performance-evaluation--benchmark-results)
12. [Repository Layout](#12-repository-layout)

---

## 1. Architectural Principles & The 80/20 Rule

Optimus eliminates complex distributed systems frameworks (Kubernetes, Ray, Slurm, Prometheus, Raft) in favor of deterministic Linux OS primitives, direct TCP sockets, Protocol Buffers, POSIX atomic file operations, and SQLite WAL transactions:

- **Controller-Initiated Probing (Zero Worker Heartbeats)**:
  Worker nodes remain strictly passive TCP listeners. They never initiate background heartbeats or broadcast unsolicited resource metrics. The controller polls workers on demand (`StatusRequest`, `ResourceRequest`, `CacheInventoryRequest`). If a worker misses 3 consecutive probes, the controller marks it `UNAVAILABLE`.
- **Authoritative Dataset & Full-File Caching**:
  The controller maintains immutable dataset versions exported read-only via standard NFS (`/srv/datasets`). Workers mount NFS at `/mnt/nfs-origin` and access data exclusively via a C FUSE3 filesystem mounted at `/mnt/dataset`. On cache miss, the complete file is retrieved, verified with streaming SHA-256, and atomically published locally. On cache hit, zero NFS traffic occurs.
- **Locality-Weighted Rank Assignment**:
  A single scheduler scores candidate workers using RAM headroom, CPU load, thermal health, disk capacity, and dataset locality (byte-weighted). Both the Python SDK and the Web Dashboard execute the identical scheduler algorithm.
- **CPU DDP via Gloo**:
  Distributed training uses PyTorch's native `DistributedDataParallel` with the Gloo CPU backend optimized for Raspberry Pi 5 ARM Cortex-A76 cores without requiring CUDA.
- **Fail-Stop Recovery**:
  If a worker crashes or drops out, the controller stops all remaining workers in that DDP group, selects replacement nodes, and relaunches the complete group restoring from the latest atomic checkpoint on shared storage.

```
+---------------------------------------------------------------------------------------+
|                              CONTROLLER (isaac.local)                                 |
|                                                                                       |
|   +-------------------+    +----------------------+    +--------------------------+   |
|   |   Optimus SDK     |    |   Node Discovery     |    |   Health & Resource      |   |
|   | (Dataset/Trainer) |    |   (Avahi / Nmap)     |    |   Probing Manager        |   |
|   +---------+---------+    +----------+-----------+    +------------+-------------+   |
|             |                         |                             |                 |
|   +---------v---------+    +----------v-----------+    +------------v-------------+   |
|   |  Dataset Manager  |    |  Locality-Aware      |    |  DDP Launcher &          |   |
|   |  (NFS Authority)  |    |  Weighted Scheduler  |    |  Job Recovery Manager    |   |
|   +-------------------+    +----------+-----------+    +--------------------------+   |
|                                       |                                               |
|                                       v                                               |
|                             [ selected_nodes.json ]                                   |
+---------------------------------------|-----------------------------------------------+
                                        | Binary TCP Framing (Protobuf)
         +------------------------------+-------------------------------+
         |                                                              |
         v                                                              v
+-------------------------------+              +--------------------------------+
|       WORKER 1 (pi02.local)   |              |       WORKER 2 (pi03.local)    |
|                               |              |                                |
|  +-------------------------+  |              |  +--------------------------+  |
|  | Optimus Worker Daemon   |  |              |  | Optimus Worker Daemon    |  |
|  | (TCP Server :45870)     |  |              |  | (TCP Server :45870)      |  |
|  +------------+------------+  |              |  +------------+-------------+  |
|               |               |              |               |                |
|  +------------v------------+  |              |  +------------v-------------+  |
|  | Training Subprocess     |  |              |  | Training Subprocess      |  |
|  | (PyTorch Gloo DDP)      |<-+--- Gloo -----+->| (PyTorch Gloo DDP)       |  |
|  +------------+------------+  |   AllReduce  |  +------------+-------------+  |
|               |               |              |               |                |
|  +------------v------------+  |              |  +------------v-------------+  |
|  | /mnt/dataset            |  |              |  | /mnt/dataset             |  |
|  | (FUSE Cache + SQLite)   |  |              |  | (FUSE Cache + SQLite)    |  |
|  +------------+------------+  |              |  +------------+-------------+  |
|               | (on miss)     |              |               | (on miss)      |
|  +------------v------------+  |              |  +------------v-------------+  |
|  | /mnt/nfs-origin (NFS)   |  |              |  | /mnt/nfs-origin (NFS)    |  |
+-------------------------------+              +--------------------------------+
```

---

## 2. Cluster Topology & Hardware Model

- **Controller Machine**:
  - Ubuntu Workstation: `isaac.local` (`192.168.10.127/24`)
  - Responsibilities: Authoritative storage, NFS server, discovery, scheduling, DDP dispatch, checkpoint verification, and dashboard.
- **Worker Nodes**:
  - Edge Nodes: Raspberry Pi 5 (Quad-core ARM Cortex-A76, 4GB/8GB RAM, Gigabit LAN)
  - Typical mDNS hostnames: `pi02.local`, `pi03.local`
  - Responsibilities: Passive TCP daemon, local SQLite cache database, C FUSE caching filesystem, training subprocess executor.

---

## 3. Binary Framing & Protocol Buffers Specification

Workers and the controller exchange serialized Protocol Buffers messages over persistent TCP connections. HTTP/REST is forbidden for node control.

### 3.1 4-Byte Big-Endian Framing
TCP is a byte stream. Every frame is preceded by an unsigned 4-byte big-endian payload length:

```
0                   1                   2                   3
0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                 Payload Length (uint32, >I)                   |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                                                               |
|             Serialized Envelope (Protocol Buffers)            |
|                     Length <= 4,194,304 bytes                 |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

The receiver validates `length <= max_message_size` (4 MiB) and uses `recv_exact` loops to prevent partial reads or stream desynchronization.

### 3.2 Envelope Structure (`src/optimus/protocol/messages.proto`)

```protobuf
message Envelope {
    uint32 protocol_version = 1;
    string message_id = 2;
    string sender_id = 3;
    uint64 timestamp_ms = 4;
    string auth_token = 5;

    oneof payload {
        StatusRequest status_request = 10;
        StatusResponse status_response = 11;
        ResourceRequest resource_request = 12;
        ResourceResponse resource_response = 13;
        CacheInventoryRequest cache_inventory_request = 14;
        CacheInventoryResponse cache_inventory_response = 15;
        CacheSummaryRequest cache_summary_request = 16;
        CacheSummaryResponse cache_summary_response = 17;
        JobSubmit job_submit = 18;
        JobResponse job_response = 19;
        JobStatusRequest job_status_request = 20;
        JobStatusResponse job_status_response = 21;
        JobStop job_stop = 22;
        ShutdownRequest shutdown_request = 23;
        ErrorResponse error_response = 24;
    }
}
```

---

## 4. Authoritative Dataset & Full-File FUSE Caching Architecture

### 4.1 Immutable Dataset Versions & Cryptographic Manifests
The controller manages datasets under `/srv/datasets/<dataset_id>/<version>/`. When prepared, the controller walks the tree and generates `manifest.json`:
```json
{
  "dataset_id": "animals",
  "version": "v1",
  "total_bytes": 496800,
  "total_files": 40,
  "files": [
    { "path": "cat/001.png", "size": 12420, "sha256": "3a7b..." },
    { "path": "dog/002.png", "size": 12420, "sha256": "8f1c..." }
  ]
}
```
Datasets are strictly immutable. Updating a dataset requires creating a new version (`v2`), preventing complex distributed cache invalidation.

### 4.2 C FUSE Filesystem Implementation (`src/optimus/cache/fuse_cache.c`)
- **Compilation**: Written in C against `libfuse3`, `libsqlite3`, and `libssl` (`OpenSSL 3+`).
- **Deferred Download on `getattr`**: `getattr()` checks the local cache file, falling back to `lstat()` on the origin NFS directory. It never downloads file content. This prevents directory traversals (`os.walk`, `ls -l`, dataset indexing) from triggering massive network storms.
- **Process-Shared Locking on Cache Miss**:
  When an application opens an uncached file:
  1. Opens `/var/cache/optimus/locks/<rel_path>.lock`.
  2. Acquires process-shared lock via `flock(fd, LOCK_EX)`.
  3. Rechecks SQLite `cached_files` (in case a concurrent thread downloaded it while waiting).
  4. Opens origin file at `/mnt/nfs-origin/<dataset_id>/<version>/<rel_path>`.
  5. Creates temporary file `/var/cache/optimus/temporary/<file>.<pid>.<rand>.downloading`.
  6. Streams file in 64 KiB chunks while computing SHA-256 on the fly.
  7. Calls `fsync(temp_fd)` and atomically renames (`rename(2)`) to `/var/cache/optimus/files/<rel_path>`.
  8. Commits record to SQLite `cached_files` and increments `registry_revision`.
  9. Releases lock (`flock(fd, LOCK_UN)`).
  10. Opens local file descriptor, saves in `fi->fh`, and serves `read()` requests directly via `pread()`.

---

## 5. Worker SQLite Cache Registry

Each worker node maintains its own local SQLite database at `/var/cache/optimus/cache.db`.

```sql
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;

CREATE TABLE cached_files (
    dataset_id TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    local_path TEXT NOT NULL,
    valid INTEGER NOT NULL,
    last_access_time REAL,
    access_count INTEGER DEFAULT 0,
    download_time REAL,
    PRIMARY KEY (dataset_id, dataset_version, relative_path)
);

CREATE TABLE cache_statistics (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cache_hits INTEGER DEFAULT 0,
    cache_misses INTEGER DEFAULT 0,
    downloaded_bytes INTEGER DEFAULT 0,
    download_failures INTEGER DEFAULT 0,
    cached_bytes INTEGER DEFAULT 0,
    cached_files INTEGER DEFAULT 0,
    registry_revision INTEGER DEFAULT 0
);
```

### 5.1 Cache Inventory Polling & Monotonic Revisions
To avoid transmitting large inventories repeatedly, each commit increments `registry_revision`. When the controller queries `CacheInventoryRequest(known_revision=N)`:
- If `known_revision == registry_revision`, the worker returns `revision_matched = True` with 0 file entries.
- If revisions differ, the worker streams paginated inventory records (`CachedFileEntry`).

---

## 6. Locality-Aware Weighted Scheduler

The scheduler runs exclusively on the controller and is shared identically by the SDK and Dashboard (`src/optimus/controller/scheduler.py`).

### 6.1 Worker Exclusion Filter
Candidates are immediately excluded if:
1. `status != AVAILABLE` or worker is unresponsive.
2. `active_jobs > 0` (node is currently executing a job).
3. `ram_available_mb < required_ram_mb` (insufficient memory headroom).
4. `cache_disk_available_bytes < (total_dataset_bytes - cached_dataset_bytes)` (insufficient disk space for missing files).

### 6.2 Weighted Scoring Formulation
Eligible workers are scored using normalized $[0, 1]$ metrics:
$$\text{Score} = 0.50 \cdot \text{locality} + 0.20 \cdot \text{ram} + 0.15 \cdot \text{cpu} + 0.10 \cdot \text{disk} + 0.05 \cdot \text{thermal}$$

Where:
- $\text{locality} = \frac{\sum \text{size of cached required files}}{\sum \text{total required dataset size}}$ (byte-weighted).
- $\text{ram} = \min\left(1.0, \frac{\text{ram\_available\_mb}}{\max(\text{eligible RAMs})}\right)$.
- $\text{cpu} = 1.0 - \left(\frac{\text{cpu\_percent}}{100.0}\right)$.
- $\text{disk} = \min\left(1.0, \frac{\text{cache\_disk\_available\_gb}}{\max(\text{eligible Disks})}\right)$.
- $\text{thermal} = \max\left(0.0, 1.0 - \frac{\max(0.0, \text{temp\_c} - 35.0)}{50.0}\right)$. If thermal telemetry is unavailable, thermal weight is omitted and remaining weights are renormalized.

### 6.3 Atomic Artifact Generation
Top $K = \text{world\_size}$ nodes are assigned ranks $0 \dots K-1$. The scheduler atomically commits `runs/<job_id>/selected_nodes.json`:
```json
{
  "job_id": "job-95620cf5",
  "dataset_id": "animals",
  "dataset_version": "v1",
  "world_size": 2,
  "selected_nodes": [
    {
      "hostname": "pi02.local",
      "port": 45870,
      "rank": 0,
      "cached_bytes": 496800,
      "required_bytes": 496800,
      "cache_locality": 1.0,
      "score": 0.9421
    },
    {
      "hostname": "pi03.local",
      "port": 45870,
      "rank": 1,
      "cached_bytes": 248400,
      "required_bytes": 496800,
      "cache_locality": 0.5,
      "score": 0.6915
    }
  ]
}
```

---

## 7. PyTorch DDP & Fail-Stop Fault Tolerance

### 7.1 Distributed Process Startup
1. The controller reads `selected_nodes.json` and issues `JobSubmit` to each node.
2. Worker daemons validate memory headroom, reserve resources, and spawn `src/optimus/training/worker_entry.py`.
3. Standard DDP environment variables are injected:
   - `RANK`: assigned rank.
   - `WORLD_SIZE`: total ranks.
   - `MASTER_ADDR`: Rank 0 hostname/IP.
   - `MASTER_PORT`: rendezvous port (29500+).

### 7.2 Atomic Checkpoints on Shared NFS
Rank 0 commits checkpoints to `/srv/checkpoints/<job_id>/checkpoint-<epoch>/`:
- `model.pt`: model weights.
- `optimizer.pt`: optimizer state.
- `rng.pt`: PyTorch CPU/CUDA random generator states.
- `metadata.json`: epoch, global step, loss, dataset version.
Written to a `.tmp` directory, flushed, and atomically renamed. `latest.json` is atomically updated, and older checkpoints are pruned (retaining the 3 newest).

### 7.3 Fail-Stop Recovery Sequence
If any worker drops out during training:
1. Controller-initiated probing detects 3 missed probes $\to$ marks node `UNAVAILABLE`.
2. Controller immediately dispatches `JobStop` to all surviving workers in that DDP group.
3. Controller verifies `restart_count <= max_restarts` (default 3).
4. Scheduler runs again, excluding the failed node and selecting available healthy nodes.
5. New `selected_nodes.json` is generated.
6. Controller reads `latest.json` from `/srv/checkpoints/<job_id>/latest.json` and relaunches the complete group with `resume_from_checkpoint=True`.
7. Workers load weights and optimizer state, resuming from epoch $E+1$.

---

## 8. Web Dashboard & REST Observability

Hosted exclusively on the controller at `http://isaac.local:8080`.

- **Minimalist Aesthetic**: Clean white background with one primary accent color (`#2563eb`).
- **REST Endpoints**:
  - `GET /`: Serves dashboard UI.
  - `GET /api/overview`: Cluster stats, active workers, cached bytes.
  - `GET /api/workers`: Live CPU %, RAM, thermals, disk, status.
  - `GET /api/cache`: Per-node cache hits, misses, downloaded bytes.
  - `POST /api/schedule`: Invokes the scheduler and returns selected nodes.
  - `GET /api/jobs`: Active and completed jobs list.
  - `POST /api/jobs/{job_id}/stop`: Gracefully terminates a running job.
  - `POST /api/probe`: Forces fresh hardware telemetry probe across cluster.

---

## 9. Developer-Facing SDK Manual

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from optimus import Optimus, Dataset, Trainer

# 1. Connect to cluster controller
cluster = Optimus(config="config.yaml")

# 2. Define authoritative dataset
dataset = Dataset(
    source="/srv/datasets/animals/v1",
    dataset_id="animals",
    version="v1"
)

# 3. Define model and optimizer
model = resnet18(weights=None, num_classes=2)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 4. Initialize managed Trainer
trainer = Trainer(
    cluster=cluster,
    model=model,
    dataset=dataset,
    optimizer=optimizer,
    epochs=10,
    workers=2,
    batch_size=16
)

# 5. Fit model across cluster
metrics = trainer.fit()

# 6. Final model weights are loaded directly onto controller
print("Trained model artifact:", metrics["final_model_path"])
```

---

## 10. Installation & Deployment Guide

### 10.1 Controller Setup (Ubuntu: `isaac.local`)

1. **System Packages**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential libfuse3-dev fuse3 libsqlite3-dev \
       protobuf-compiler nmap avahi-utils nfs-kernel-server python3-pip
   ```

2. **Python Dependencies**:
   ```bash
   pip install -e .
   ```

3. **Storage & NFS Server Configuration**:
   ```bash
   sudo mkdir -p /srv/datasets /srv/checkpoints /var/cache/optimus
   sudo chown -R $USER:$USER /srv/datasets /srv/checkpoints /var/cache/optimus
   ```
   Add exports to `/etc/exports`:
   ```text
   /srv/datasets    192.168.10.0/24(ro,sync,no_subtree_check,no_root_squash)
   /srv/checkpoints 192.168.10.0/24(rw,sync,no_subtree_check,no_root_squash)
   ```
   Apply exports:
   ```bash
   sudo exportfs -ra
   sudo systemctl restart nfs-kernel-server
   ```

4. **Launch Web Dashboard**:
   ```bash
   uvicorn optimus.dashboard.app:app --host 0.0.0.0 --port 8080
   ```

### 10.2 Worker Node Setup (Raspberry Pi 5)

1. **System Packages**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y build-essential libfuse3-dev fuse3 libsqlite3-dev \
       libssl-dev nfs-common avahi-daemon python3-pip python3-protobuf python3-yaml
   ```

2. **Mount Directories & NFS Mount**:
   ```bash
   sudo mkdir -p /mnt/nfs-origin /mnt/dataset /var/cache/optimus /srv/checkpoints
   sudo chown -R $USER:$USER /mnt/dataset /var/cache/optimus /srv/checkpoints

   # Mount controller authoritative exports
   sudo mount -t nfs -o ro,soft,timeo=30 isaac.local:/srv/datasets /mnt/nfs-origin
   sudo mount -t nfs -o rw,soft,timeo=30 isaac.local:/srv/checkpoints /srv/checkpoints
   ```

3. **Build FUSE Cache Binary**:
   ```bash
   cd ~/OptimusDL/src/optimus/cache
   make clean && make
   ```

4. **Install Optimus Package**:
   ```bash
   cd ~/OptimusDL
   pip3 install -e . --no-deps --break-system-packages
   ```

5. **Run Worker Daemon as a Systemd Service (Recommended)**:
   Create `/etc/systemd/system/optimus-worker.service`:
   ```ini
   [Unit]
   Description=Optimus Worker Daemon
   After=network.target

   [Service]
   Type=simple
   User=pi
   WorkingDirectory=/home/pi/OptimusDL
   ExecStart=/usr/bin/python3 -m optimus.worker.daemon --config /home/pi/OptimusDL/config.yaml
   Restart=always
   RestartSec=3
   StandardOutput=journal
   StandardError=journal

   [Install]
   WantedBy=multi-user.target
   ```
   Enable and start the service:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable --now optimus-worker
   sudo systemctl status optimus-worker
   ```

   *(Alternatively, run in the foreground for debugging: `python3 -m optimus.worker.daemon --config config.yaml`)*

---

### 10.3 Cluster Verification & Execution

1. **Probe Cluster Status & Worker Telemetry**:
   ```bash
   python cluster.py --config config.yaml
   # Or continuous watch mode:
   python cluster.py --config config.yaml --watch
   ```

2. **Launch Observability Web Dashboard**:
   ```bash
   uvicorn optimus.dashboard.app:app --host 0.0.0.0 --port 8080
   ```
   Open [http://localhost:8080](http://localhost:8080) to inspect worker availability, CPU/RAM/thermals, and cache inventory.

3. **Run Distributed ResNet-18 Training Example**:
   ```bash
   python example/train_resnet.py
   ```

4. **Run System Test Suite**:
   ```bash
   pytest test/ -v
   ```

---

## 11. Performance Evaluation & Benchmark Results

Run benchmark evaluation:
```bash
python3 example/evaluate_performance.py
```

### 11.1 Benchmark Measurements (`runs/evaluation_results.csv`)

| Scenario | Elapsed Time | Throughput | Cached Files | Speedup vs Cold |
|---|---|---|---|---|
| **Baseline Origin Access** | 0.0015s | 4099.20 MB/s | 0 | 37.8x |
| **Optimus Cold Cache (Miss)** | 0.0567s | 110.16 MB/s | 100 | 1.00x (Baseline) |
| **Optimus Warm Cache (Hit)** | 0.0583s | 107.24 MB/s | 100 | **Zero NFS Reads** |

### 11.2 DDP Training Scaling (ResNet-18 on Gloo CPU)
- **1 Worker Throughput**: 70.9 samples/sec
- **2 Workers Throughput**: 129.0 samples/sec
- **Scaling Efficiency**: **91.0%**

---

## 12. Repository Layout

```
optimus/
|-- cluster.py                   # Master cluster CLI (validation & watch)
|-- config.yaml                  # Cluster configuration
|-- requirements.txt             # Python dependencies
|-- pyproject.toml               # Package build metadata
|-- README.md                    # System documentation
|
|-- src/
|   |-- optimus/
|       |-- __init__.py          # Top-level SDK exports
|       |-- cluster.py           # Configuration schema
|       |-- sdk.py               # Developer Optimus class
|       |-- dataset.py           # Dataset & manifest generator
|       |-- trainer.py           # Managed Trainer API
|       |
|       |-- protocol/
|       |   |-- messages.proto   # Protocol Buffers schema
|       |   |-- framing.py       # 4-byte big-endian TCP framing
|       |   |-- generated/       # Generated protobuf classes
|       |
|       |-- controller/
|       |   |-- communication.py # Persistent TCP client pool
|       |   |-- probing.py       # Controller-initiated health prober
|       |   |-- discovery.py     # Avahi mDNS + Nmap scanner
|       |   |-- dataset_manager.py # NFS dataset publisher
|       |   |-- cache_registry.py  # Cluster cache coordinator
|       |   |-- scheduler.py     # Weighted locality scheduler
|       |   |-- launcher.py      # DDP remote rank launcher
|       |   |-- job_manager.py   # Job lifecycle & fail-stop recovery
|       |   |-- checkpoint.py    # Atomic checkpoint manager
|       |
|       |-- worker/
|       |   |-- server.py        # Passive TCP listener
|       |   |-- daemon.py        # Worker daemon process
|       |   |-- resource_info.py # CPU, RAM, thermal reader
|       |   |-- executor.py      # Subprocess DDP runner
|       |
|       |-- cache/
|       |   |-- fuse_cache.c     # Full-file FUSE3 caching in C
|       |   |-- Makefile         # FUSE cache compilation
|       |   |-- registry.py      # SQLite cache database
|       |   |-- manager.py       # Cache reconciliation
|       |
|       |-- training/
|       |   |-- ddp.py           # Gloo process group setup
|       |   |-- sampler.py       # DistributedSampler wrapper
|       |   |-- worker_entry.py  # Worker DDP training entrypoint
|       |
|       |-- dashboard/
|           |-- app.py           # FastAPI application
|           |-- templates/       # Clean HTML UI
|           |-- static/          # CSS & JavaScript
|
|-- example/
|   |-- example_dataset.py       # Synthetic dataset builder
|   |-- train_resnet.py          # End-to-end ResNet training
|   |-- evaluate_performance.py  # I/O & scaling benchmark
|   |-- config.yaml              # Local test configuration
|
|-- test/
    |-- test_protocol.py         # Framing, envelopes, size checks
    |-- test_probing.py          # Probing, threshold, recovery
    |-- test_resource.py         # Hardware telemetry probing
    |-- test_discovery.py        # Avahi & Nmap discovery
    |-- test_cache.py            # SQLite registry & FUSE cache
    |-- test_cache_concurrency.py# Process flock locking
    |-- test_scheduler.py        # Weighted locality scoring
    |-- test_checkpoint.py       # Atomic checkpoint commit/restore
    |-- test_training.py         # Gloo DDP & executor lifecycle
```
