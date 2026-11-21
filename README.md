# Optimus: Lightweight Distributed ML Training System

Optimus is an ultra-lightweight, resilient distributed machine learning training platform engineered specifically for edge clusters composed of multiple Raspberry Pi 5 nodes coordinated by a designated Linux controller workstation (`isaac.local`).

Optimus abstracts distributed infrastructure so ML developers write ordinary PyTorch training scripts using the custom Optimus SDK. The backend transparently orchestrates worker discovery via Nmap and Avahi, single-roundtrip hardware telemetry probing, pure-Python atomic caching with streaming SHA-256 verification over NFS, locality-aware weighted scheduling, PyTorch DistributedDataParallel (DDP) over Gloo, atomic checkpointing with per-node verification, and fail-stop recovery.

---

## Table of Contents

1. [Architectural Principles & Streamlined Design](#1-architectural-principles--streamlined-design)
2. [Cluster Topology & Hardware Model](#2-cluster-topology--hardware-model)
3. [optimusd REST Control-Plane API](#3-optimusd-rest-control-plane-api)
4. [Authoritative Datasets & Pure-Python Atomic Caching](#4-authoritative-datasets--pure-python-atomic-caching)
5. [Worker SQLite Cache Registry](#5-worker-sqlite-cache-registry)
6. [Locality-Aware Weighted Scheduler](#6-locality-aware-weighted-scheduler)
7. [PyTorch Gloo DDP & Fail-Stop Fault Tolerance](#7-pytorch-gloo-ddp--fail-stop-fault-tolerance)
8. [Web Dashboard & REST Observability](#8-web-dashboard--rest-observability)
9. [Developer-Facing SDK Manual](#9-developer-facing-sdk-manual)
10. [Installation & Deployment Guide](#10-installation--deployment-guide)
11. [Performance Evaluation & Benchmark Results](#11-performance-evaluation--benchmark-results)
12. [Repository Layout](#12-repository-layout)

---

## 1. Architectural Principles & Streamlined Design

Optimus eliminates complex distributed systems frameworks (Kubernetes, Ray, Slurm, Prometheus, Raft) and heavyweight external binary dependencies in favor of deterministic Linux OS primitives, a small stdlib-only REST/JSON control API, POSIX atomic file operations, and SQLite WAL transactions:

- **Controller-Initiated Probing (Zero Worker Heartbeats)**:
  Worker nodes run `optimusd`, a strictly passive HTTP server on port `45870`. They never initiate background heartbeats or broadcast unsolicited resource metrics. The controller polls workers on demand (`GET /v1/status`, `/v1/resources`, `/v1/cache/inventory`, `/v1/checkpoints`) in consolidated single-request telemetry exchanges. If a worker misses 3 consecutive probes, the controller marks it `UNAVAILABLE`.
- **Pure-Python Atomic Caching (No C/FUSE Overhead)**:
  The controller maintains immutable dataset versions exported read-only via standard NFS (`/srv/datasets`). Workers mount NFS at `/mnt/nfs-origin` and access data via an atomic, multi-process safe Python cache engine (`fcntl.flock`). On cache miss, the file is streamed in 64 KiB chunks, verified with streaming SHA-256, and atomically committed to `/var/cache/optimus/`. On cache hit, reads are served locally with zero NFS traffic.
- **Locality-Weighted Rank Assignment**:
  A single scheduler scores candidate workers using RAM headroom, CPU load, thermal health, disk capacity, and dataset locality (byte-weighted). Both the Python SDK and the Web Dashboard execute the identical scheduler algorithm.
- **CPU DDP via Gloo with Dynamic Interface Binding**:
  Distributed training uses PyTorch's native `DistributedDataParallel` with the Gloo CPU backend optimized for Raspberry Pi 5 ARM Cortex-A76 cores without requiring CUDA. Optimus automatically detects the active LAN interface (`eth0`) and resolves rendezvous IPs dynamically, preventing multi-node Debian loopback (`127.0.1.1`) binding errors.
- **Fail-Stop Recovery**:
  If a worker crashes or drops out, the controller stops all remaining workers in that DDP group, selects replacement nodes, and relaunches the complete group restoring from the latest atomic checkpoint on shared storage.
- **Saved Checkpoint Observability Across Nodes**:
  Rank 0 workers commit atomic checkpoints at each epoch to shared storage (`/srv/checkpoints`). The controller and web dashboard actively verify checkpoint presence and mount status across every individual node (`pi02.local`, `pi03.local`, `isaac.local`).

```
+---------------------------------------------------------------------------------------+
|                              CONTROLLER (isaac.local)                                 |
|                                                                                       |
|   +-------------------+    +----------------------+    +--------------------------+   |
|   |   Optimus SDK     |    |   Node Discovery     |    |   Consolidated Probing   |   |
|   | (Dataset/Trainer) |    |   (Avahi / Nmap)     |    |   (Hardware Telemetry)   |   |
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
                                        | REST / JSON over HTTP :45870 (control plane)
         +------------------------------+-------------------------------+
         |                                                              |
         v                                                              v
+-------------------------------+              +--------------------------------+
|       WORKER 1 (pi02.local)   |              |       WORKER 2 (pi03.local)    |
|                               |              |                                |
|  +-------------------------+  |              |  +--------------------------+  |
|  | optimusd                |  |              |  | optimusd                 |  |
|  | (REST API :45870)       |  |              |  | (REST API :45870)        |  |
|  +------------+------------+  |              |  +------------+-------------+  |
|               |               |              |               |                |
|  +------------v------------+  |              |  +------------v-------------+  |
|  | Training Subprocess     |  |              |  | Training Subprocess      |  |
|  | (PyTorch Gloo DDP)      |<-+--- Gloo -----+->| (PyTorch Gloo DDP)       |  |
|  +------------+------------+  |  (eth0 Mesh) |  +------------+-------------+  |
|               |               |              |               |                |
|  +------------v------------+  |              |  +------------v-------------+  |
|  | /var/cache/optimus      |  |              |  | /var/cache/optimus       |  |
|  | (Pure-Python flock)     |  |              |  | (Pure-Python flock)      |  |
|  +------------+------------+  |              |  +------------+-------------+  |
|               | (on miss)     |              |               | (on miss)      |
|  +------------v------------+  |              |  +------------v-------------+  |
|  | /mnt/nfs-origin (NFS)   |  |              |  | /mnt/nfs-origin (NFS)    |  |
|  +-------------------------+  |              |  +--------------------------+  |
|  | /srv/checkpoints (NFS)  |  |              |  | /srv/checkpoints (NFS)   |  |
|  +-------------------------+  |              |  +--------------------------+  |
+-------------------------------+              +--------------------------------+
```

---

## 2. Cluster Topology & Hardware Model

- **Controller Workstation**:
  - Ubuntu Workstation: `isaac.local` (`192.168.10.127/24`)
  - Responsibilities: Authoritative dataset storage (`/srv/datasets`), NFS server, discovery, scheduling, DDP dispatch, checkpoint verification, and dashboard.
- **Worker Nodes**:
  - Edge Nodes: Raspberry Pi 5 (Quad-core ARM Cortex-A76, 8GB RAM, Gigabit LAN)
  - Typical mDNS hostnames: `pi02.local` (`192.168.10.150`), `pi03.local` (`192.168.10.101`)
  - Responsibilities: Passive REST daemon `optimusd` (`optimusd.service`), local SQLite cache database, atomic cache manager, training subprocess executor.

---

## 3. optimusd REST Control-Plane API

Every worker runs **`optimusd`**, the Optimus worker daemon. The controller talks to it over a small REST API: plain HTTP/1.1 with JSON bodies on port `45870`, served and consumed with the Python standard library only (`http.server` / `http.client`, no extra dependencies). The API is defined in `src/optimus/protocol/rest.py`; request/response bodies are the dataclasses in `src/optimus/protocol/messages.py`.

This API is **control plane only**. Gradient all-reduce and other training traffic still flows directly between ranks over PyTorch Gloo (see section 7) and never goes through `optimusd`.

### 3.1 Endpoints

| Method | Path | Body / query | Response |
|---|---|---|---|
| `GET` | `/v1/status` | | `StatusResponse` (availability, active jobs, uptime) |
| `GET` | `/v1/resources` | | `ResourceResponse` (CPU, RAM, cache disk, SoC thermals) |
| `GET` | `/v1/cache/summary` | `?dataset_id=&dataset_version=` | `CacheSummaryResponse` (hits, misses, bytes, registry revision) |
| `GET` | `/v1/cache/inventory` | `?dataset_id=&dataset_version=&known_revision=&page=&page_size=` | `CacheInventoryResponse` (paginated files with SHA-256) |
| `GET` | `/v1/checkpoints` | `?checkpoint_dir=&job_id=` | `CheckpointListResponse` (saved checkpoints, disk space, mount status) |
| `POST` | `/v1/jobs` | `JobSubmit` | `201` `JobResponse` if accepted, `409` `JobResponse` if rejected |
| `GET` | `/v1/jobs/{job_id}` | | `JobStatusResponse` (state, epoch, step, loss, throughput) |
| `POST` | `/v1/jobs/{job_id}/stop` | `JobStop` (`reason`, `graceful`) | `StatusResponse` |
| `POST` | `/v1/shutdown` | `ShutdownRequest` (`reason`, `force`) | `202` `StatusResponse`, then the daemon exits |

Enum fields (`status`) are sent as integers, see `WorkerState` and `JobState` in `messages.py`. Unknown JSON fields are ignored, so new optional fields can be added without breaking older peers.

### 3.2 Headers, auth and errors

- **Auth**: when `cluster.secret` is set in `config.yaml`, every request must send `Authorization: Bearer <secret>` (compared in constant time). Otherwise the daemon answers `401 UNAUTHORIZED`.
- **Versioning**: clients send `X-Optimus-Protocol-Version: 1`; a different value gets `400 VERSION_MISMATCH`. Responses carry the same header. Paths are also versioned (`/v1`).
- **Size limit**: bodies larger than `communication.max_message_size` (4 MiB default) are refused with `413 MESSAGE_TOO_LARGE`.
- **Errors**: any 4xx/5xx carries an `ErrorResponse` body, e.g. `{"error_code": "NOT_FOUND", "error_message": "No route for GET /v1/nope"}`. Codes: `BAD_REQUEST`, `UNAUTHORIZED`, `VERSION_MISMATCH`, `NOT_FOUND`, `METHOD_NOT_ALLOWED`, `MESSAGE_TOO_LARGE`, `INTERNAL_ERROR`.

### 3.3 Trying it by hand

```bash
curl -H "Authorization: Bearer $SECRET" http://pi02.local:45870/v1/status
curl -H "Authorization: Bearer $SECRET" "http://pi02.local:45870/v1/checkpoints?checkpoint_dir=/srv/checkpoints"
```

From Python, `optimus.controller.communication.WorkerClient` wraps each endpoint (`get_status()`, `get_resources()`, `get_cache_summary()`, `get_cache_inventory()`, `list_checkpoints()`, `submit_job()`, `get_job_status()`, `stop_job()`, `shutdown()`). Extra routes can be added on the worker with `WorkerServer.register_route("GET", "/v1/foo/{id}", handler)`.

---

## 4. Authoritative Datasets & Pure-Python Atomic Caching

### 4.1 Immutable Dataset Versions & Cryptographic Manifests
The controller manages datasets under `/srv/datasets/<dataset_id>/<version>/`. When prepared, the controller walks the tree and generates `manifest.json`:
```json
{
  "dataset_id": "animals",
  "version": "v1",
  "total_bytes": 1242000,
  "total_files": 100,
  "files": [
    { "path": "cat/img_0000.png", "size": 12420, "sha256": "3a7b..." },
    { "path": "dog/img_0000.png", "size": 12420, "sha256": "8f1c..." }
  ]
}
```
Datasets are strictly immutable. Updating a dataset requires creating a new version (`v2`), preventing complex distributed cache invalidation bugs.

### 4.2 Multi-Process Atomic Cache Engine (`src/optimus/cache/manager.py`)
Replaces C FUSE filesystems with a pure-Python atomic caching layer:
1. **Concurrency Control**: Acquires an exclusive lock on `/var/cache/optimus/locks/<rel_path>.lock` using `fcntl.flock(fd, LOCK_EX)`.
2. **Double-Checked Locking**: Checks the local SQLite registry to verify if another process downloaded the file while waiting.
3. **Streaming Verification**: Streams file from NFS `/mnt/nfs-origin/` into `/var/cache/optimus/temporary/<file>.<pid>.downloading` in 64 KiB chunks while computing SHA-256 on the fly.
4. **Digest Validation**: If SHA-256 matches the manifest, calls `os.fsync()` and atomically renames the file to `/var/cache/optimus/files/<rel_path>`.
5. **Registry Update**: Records the entry into SQLite WAL database and increments `registry_revision`.
6. **Subsequent Reads**: Process opens local cache file descriptor with zero NFS network traffic.

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
To avoid transmitting large file inventories repeatedly, each commit increments `registry_revision`. When the controller queries `CacheInventoryRequest(known_revision=N)`:
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
- $\text{thermal} = \max\left(0.0, 1.0 - \frac{\max(0.0, \text{temp\_c} - 35.0)}{50.0}\right)$.

### 6.3 Atomic Artifact Generation
Top $K = \text{world\_size}$ nodes are assigned ranks $0 \dots K-1$. The scheduler commits `runs/<job_id>/selected_nodes.json`:
```json
{
  "job_id": "job-4a414015",
  "dataset_id": "animals",
  "dataset_version": "v1",
  "world_size": 2,
  "selected_nodes": [
    {
      "hostname": "pi02.local",
      "port": 45870,
      "worker_id": "worker-pi02",
      "rank": 0,
      "score": 0.4827
    },
    {
      "hostname": "pi03.local",
      "port": 45870,
      "worker_id": "worker-pi03",
      "rank": 1,
      "score": 0.4493
    }
  ]
}
```

---

## 7. PyTorch Gloo DDP & Fail-Stop Fault Tolerance

### 7.1 Distributed Process Startup & Dynamic Gloo Binding
1. The controller reads `selected_nodes.json` and issues `JobSubmit` to each node.
2. Worker daemons validate memory headroom, reserve resources, and launch `optimus.training.worker_entry`.
3. Optimus automatically detects the active non-loopback network interface (e.g. `eth0`) and resolves `MASTER_ADDR` to the direct IPv4 address (`192.168.10.150:29500`). This completely avoids Debian `/etc/hosts` loopback binding issues (`127.0.1.1`).
4. Injected environment variables:
   - `RANK`: assigned rank.
   - `WORLD_SIZE`: total ranks.
   - `MASTER_ADDR`: Rank 0 resolved IP address.
   - `MASTER_PORT`: rendezvous port (29500+).
   - `GLOO_SOCKET_IFNAME`: active network interface (`eth0`).

### 7.2 Atomic Checkpoints on Shared Storage
Rank 0 commits checkpoints to `/srv/checkpoints/<job_id>/checkpoint-<epoch>/`:
- `model.pt`: model weights state dictionary.
- `optimizer.pt`: optimizer state dictionary.
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

Hosted on the controller at `http://isaac.local:8080`.

- **UI Stack**: **Alpine.js** (declarative state and rendering, one small component in `static/app.js`) on **Pico CSS**, themed to the **IBM Carbon** design language: 48px dark shell header, 256px side navigation, square-cornered tiles, Carbon data-table headers and pill tags, 8px spacing grid. Fonts are **IBM Plex Sans** (UI) and **IBM Plex Mono** (IDs, paths, metrics). Alpine, Pico and the fonts are vendored in `static/vendor/`, so the dashboard works on a LAN with no internet. There is no hand-written DOM or boilerplate JavaScript.
- **Three-color palette** (`static/theme.css`, enforced by a test):
  - Paper `#f4f4f4` and ink `#161616` are the two majority colors (surfaces, text, borders, header). Muted text, borders and table headers are mixes of just these two.
  - Primary blue `#0f62fe` is reserved for important parts: key metrics, primary actions, healthy/verified tags, the active nav item, focus rings.
- **Dedicated Dashboard Views**:
  - **Overview**: Real-time worker availability counts, active jobs, cached bytes, and saved checkpoints summary card.
  - **LAN Topology Discovery**: Scans subnet via `nmap -sn` and resolves reverse mDNS names via `avahi-resolve -a` to classify gateways, controllers, active workers, and LAN devices.
  - **Cluster Workers**: Detailed per-worker telemetry (CPU %, available RAM, disk capacity, thermals, uptime).
  - **Locality Scheduler**: Form to test rank allocation across nodes for any dataset.
  - **Distributed Training Jobs**: Monitors active PyTorch DDP runs, epochs, steps, loss, and restarts.
  - **Dataset Cache**: Local cache inventory and SHA-256 verification stats.
  - **Saved Checkpoints**: Node storage & mount verification strip (`isaac.local`, `pi02.local`, `pi03.local`), node filter bar, checkpoint inventory table (with origin Rank 0, verified nodes pills, loss, size), and interactive metadata JSON inspector.
- **REST Endpoints**:
  - `GET /`: Serves the dashboard web app.
  - `GET /api/overview`: Cluster stats, active workers, cached bytes, total checkpoints.
  - `GET /api/workers`: Live CPU %, RAM, thermals, disk, status.
  - `GET /api/discovery`: Returns discovered LAN topology hosts.
  - `POST /api/discovery/scan`: Triggers an active Nmap subnet sweep and Avahi mDNS resolution.
  - `GET /api/cache`: Per-node cache hits, misses, downloaded bytes.
  - `POST /api/schedule`: Invokes the scheduler and returns selected nodes.
  - `GET /api/jobs`: Active and completed jobs list.
  - `POST /api/jobs/{job_id}/stop`: Gracefully terminates a running job.
  - `GET /api/checkpoints`: Scans cluster and queries each worker's `GET /v1/checkpoints` for verified saved checkpoints.
  - `GET /api/checkpoints/{job_id}/{checkpoint_name}/metadata`: Returns raw metadata JSON.
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

# Check worker health and telemetry
print(cluster.probe())

# 2. Define authoritative dataset
dataset = Dataset(
    source="/srv/datasets/animals/v1",
    dataset_id="animals",
    version="v1"
)

# 3. Define model and optimizer
model = resnet18(weights=None, num_classes=2)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# 4. Initialize managed Trainer. Workers rebuild the model by importing
#    model_factory ("module:function"); it must be importable on every worker.
trainer = Trainer(
    cluster=cluster,
    model=model,
    dataset=dataset,
    optimizer=optimizer,
    model_factory="torchvision.models:resnet18",
    model_kwargs={"num_classes": 2},
    epochs=3,
    workers=2,
    batch_size=16
)

# 5. Fit model across Raspberry Pi cluster
metrics = trainer.fit()

# 6. Final model weights are loaded directly onto controller
print("Trained model artifact:", metrics["final_model_path"])
if os.path.isfile(metrics["final_model_path"]):
    model.load_state_dict(torch.load(metrics["final_model_path"], map_location="cpu"))
    print("Final model state successfully loaded into local model!")
```

### 9.1 Model & Dataset Factories

Workers never receive pickled objects or code over the network. `Trainer` sends **import paths**
(`"package.module:function"`) plus JSON kwargs, and each worker imports and calls them locally
(`src/optimus/training/factories.py`). Adding a new model or dataset therefore never requires
editing `src/optimus`: put the factories in any module importable on the workers (e.g. under the
repo root, which is optimusd's working directory) and pass their paths.

| Argument | Contract | Default |
|---|---|---|
| `model_factory` | `fn(**model_kwargs) -> nn.Module`; `num_classes` is filled in if the function accepts it | required unless `model` is a torchvision ResNet (deprecated) or has `model_name` |
| `dataset_factory` | `fn(data_dir, **dataset_kwargs) -> torch Dataset`; `data_dir` is the worker's dataset copy or `None` | ImageFolder if class folders exist, else synthetic tensors |
| `model_kwargs` / `dataset_kwargs` | JSON-serializable dicts | `{}` |

Factories may also be passed as module-level callables; `Trainer` converts them to import paths
(functions defined in the `__main__` script are rejected, since workers cannot import them).
See `example/cifar10.py` and `example/train_cifar10.py` for a complete custom model + dataset.

---

## 10. Installation & Deployment Guide

### 10.1 Controller Setup (Ubuntu: `isaac.local`)

1. **System Packages**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y nmap avahi-utils nfs-kernel-server python3-pip
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

---

### 10.2 Worker Node Setup (Raspberry Pi 5)

1. **System Packages**:
   ```bash
   sudo apt-get update
   sudo apt-get install -y nfs-common avahi-daemon python3-pip python3-yaml
   ```

2. **Mount Directories & NFS Mounts**:
   ```bash
   sudo mkdir -p /mnt/nfs-origin /var/cache/optimus /srv/checkpoints
   sudo chown -R pi:pi /var/cache/optimus /srv/checkpoints

   # Mount controller authoritative exports
   sudo mount -t nfs -o ro,soft,timeo=30 192.168.10.127:/srv/datasets /mnt/nfs-origin
   sudo mount -t nfs -o rw,soft,timeo=30 192.168.10.127:/srv/checkpoints /srv/checkpoints
   ```
   *(To persist mounts across reboots, add them to `/etc/fstab`)*.

3. **Install Optimus Package**:
   ```bash
   cd ~/OptimusDL
   pip3 install -e . --no-deps --break-system-packages
   ```

4. **Run optimusd via Systemd**:
   A ready-made unit lives at `deploy/optimusd.service` (`sudo cp deploy/optimusd.service /etc/systemd/system/`). It contains:
   ```ini
   [Unit]
   Description=optimusd - Optimus worker daemon (REST control-plane API)
   After=network-online.target
   Wants=network-online.target

   [Service]
   Type=simple
   User=pi
   WorkingDirectory=/home/pi/OptimusDL
   ExecStart=/usr/bin/python3 -m optimus.worker.optimusd --config /home/pi/OptimusDL/config.yaml
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
   sudo systemctl enable --now optimusd
   sudo systemctl status optimusd
   ```
   For a quick foreground run without systemd, use the `optimusd` console script installed by `pip install -e .` (`optimusd --config config.yaml`). If you previously installed the old `optimus-worker.service`, disable and remove it first: `sudo systemctl disable --now optimus-worker`.

---

### 10.3 Cluster Verification & Execution

1. **Generate Sample Image Dataset**:
   ```bash
   python example/example_dataset.py --output /srv/datasets/animals/v1 --images-per-class 50
   ```

2. **Run Distributed ResNet-18 Training Example**:
   ```bash
   python example/train_resnet.py
   ```

3. **Launch Observability Web Dashboard**:
   ```bash
   uvicorn optimus.dashboard.app:app --host 0.0.0.0 --port 8080
   ```
   Open **http://localhost:8080** in your browser to inspect worker availability, hardware telemetry, LAN topology, and saved checkpoints.

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
|-- config.yaml                  # Cluster configuration
|-- requirements.txt             # Python dependencies
|-- pyproject.toml               # Package build metadata
|-- README.md                    # System documentation
|-- deploy/optimusd.service      # systemd unit for the optimusd worker daemon
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
|       |   |-- messages.py      # Pure-Python message dataclasses (JSON bodies)
|       |   |-- rest.py          # REST API routes, headers, JSON codec
|       |   |-- generated/       # Backward-compatibility stub
|       |
|       |-- controller/
|       |   |-- communication.py # REST client for optimusd workers
|       |   |-- probing.py       # Single-roundtrip health & hardware prober
|       |   |-- discovery.py     # Nmap subnet sweep & Avahi mDNS LAN scanner
|       |   |-- dataset_manager.py # NFS dataset publisher & manifest verifier
|       |   |-- cache_registry.py  # Cluster cache coordinator
|       |   |-- scheduler.py     # Weighted locality scheduler
|       |   |-- launcher.py      # DDP remote rank launcher
|       |   |-- job_manager.py   # Job lifecycle & fail-stop recovery
|       |   |-- checkpoint.py    # Atomic checkpoint manager
|       |
|       |-- worker/
|       |   |-- server.py        # optimusd REST server (http.server) & checkpoint reporter
|       |   |-- optimusd.py      # optimusd daemon process & CLI entry point
|       |   |-- resource_info.py # CPU, RAM, thermal reader
|       |   |-- executor.py      # Subprocess DDP runner with dynamic Gloo binding
|       |
|       |-- cache/
|       |   |-- manager.py       # Pure-Python atomic cache engine (fcntl.flock)
|       |   |-- registry.py      # SQLite cache database (WAL mode)
|       |
|       |-- training/
|       |   |-- ddp.py           # Gloo process group setup & auto interface binding
|       |   |-- sampler.py       # DistributedSampler wrapper
|       |   |-- worker_entry.py  # Worker DDP training entrypoint
|       |   |-- factories.py     # Import-path model/dataset factory resolution
|       |
|       |-- dashboard/
|           |-- app.py           # FastAPI application & checkpoint verification API
|           |-- templates/       # Alpine.js + Pico CSS single-page markup
|           |-- static/          # theme.css (Carbon tokens), app.js (Alpine component), vendor/ (Alpine, Pico, IBM Plex)
|
|-- example/
|   |-- example_dataset.py       # Synthetic dataset builder
|   |-- train_resnet.py          # End-to-end ResNet training
|   |-- cifar10.py               # CIFAR-10 ConvNet + dataset factories
|   |-- train_cifar10.py         # CIFAR-10 training via custom factories
|   |-- evaluate_performance.py  # I/O & scaling benchmark
|   |-- config.yaml              # Local test configuration
|
|-- test/
    |-- test_protocol.py         # REST API routing, auth, errors, size checks
    |-- test_probing.py          # Probing, threshold, recovery
    |-- test_resource.py         # Hardware telemetry probing
    |-- test_discovery.py        # Avahi & Nmap discovery
    |-- test_cache.py            # SQLite registry & pure-Python atomic cache
    |-- test_cache_concurrency.py# Process flock locking
    |-- test_scheduler.py        # Weighted locality scoring
    |-- test_checkpoint.py       # Atomic checkpoint commit/restore
    |-- test_training.py         # Gloo DDP & executor lifecycle
    |-- test_dashboard.py        # Web dashboard & checkpoint endpoints
```
