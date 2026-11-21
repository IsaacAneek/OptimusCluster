<div align="center">

<img src="assets/logo.svg" alt="Optimus logo" width="140" />

# Optimus

### Turn a shelf of Raspberry Pis into one training machine.

**Write normal PyTorch. Point it at your cluster. Optimus handles the rest: finding nodes, shipping data, running DDP, surviving crashes.**

<br/>

![Python](https://img.shields.io/badge/python-3.9%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-DDP%20%2B%20Gloo-EE4C2C?logo=pytorch&logoColor=white)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-5-C51A4A?logo=raspberrypi&logoColor=white)
![Control plane](https://img.shields.io/badge/control%20plane-REST%2FJSON-0f62fe)
![Dependencies](https://img.shields.io/badge/worker%20daemon-stdlib%20only-success)
![Tests](https://img.shields.io/badge/tests-71%20passing-brightgreen)
![Status](https://img.shields.io/badge/status-MVP%20prototype-orange)

</div>

---

<div align="center">

*No Kubernetes. No Ray. No Slurm. No GPU.*
*Just a controller, a few cheap boards, and a Python script.*

</div>

<br/>

## Why Optimus?

Training across several small machines usually means learning a cluster scheduler first. Optimus goes the other way. The controller is a Python library you import, and each worker runs one tiny daemon, `optimusd`, that speaks plain HTTP and JSON. Distributed training itself is stock PyTorch `DistributedDataParallel` over Gloo, so your model code stays ordinary.

What you get:

- **A small SDK**: `Optimus`, `Dataset`, `Trainer`. You describe the job, call `fit()`, and get weights back.
- **Locality-aware scheduling**: ranks go to the healthiest nodes, preferring ones that already hold your data.
- **Fail-stop recovery**: if a node drops out mid-run, the group is stopped, rescheduled and resumed from the last atomic checkpoint.
- **A live dashboard**: worker health, thermals, LAN topology, jobs and checkpoints in the browser.
- **Honest failures**: a worker with no dataset fails loudly with the paths it checked, and real tracebacks come back to your terminal.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Installation](#installation)
3. [Usage](#usage)
4. [The control-plane API](#the-control-plane-api)
5. [Scheduling](#scheduling)
6. [Training, checkpoints and recovery](#training-checkpoints-and-recovery)
7. [Dataset caching](#dataset-caching)
8. [Dashboard](#dashboard)
9. [Configuration](#configuration)
10. [Known limitations](#known-limitations)
11. [Repository layout](#repository-layout)
12. [Development and tests](#development-and-tests)

---

## How it works

Two roles. The **controller** is your workstation, where your script runs. A **worker** is any Pi running `optimusd`.

```
                         CONTROLLER  (your workstation)
   +-------------------------------------------------------------------+
   |  your script  ->  Optimus SDK  (Dataset / Trainer)                |
   |                      |                                            |
   |      Discovery -> Probing -> Scheduler -> Launcher -> JobManager   |
   +-----------------------------|-------------------------------------+
                                 |  REST / JSON over HTTP :45870
              +------------------+------------------+
              v                                     v
   +------------------------+            +------------------------+
   | WORKER  node 1         |            | WORKER  node 2         |
   |  optimusd (passive)    |            |  optimusd (passive)    |
   |    +- training process |<== Gloo ==>|    +- training process |
   |  /mnt/nfs-origin (data)|            |  /mnt/nfs-origin (data)|
   |  /srv/checkpoints      |            |  /srv/checkpoints      |
   +------------------------+            +------------------------+
```

### The life of a `fit()` call

1. **Describe the data.** `Dataset(...)` hashes every file into an immutable, versioned manifest on the controller.
2. **Probe the workers.** The controller asks each `optimusd` for status, CPU, RAM, disk, temperature and cache contents in one request. Workers never send heartbeats on their own. A node that misses 3 probes in a row is marked `UNAVAILABLE`.
3. **Schedule.** Unhealthy or busy nodes are filtered out. The rest are scored and the top *N* become ranks `0..N-1`.
4. **Launch.** The controller sends each chosen worker a job. `optimusd` starts `optimus.training.worker_entry`, which finds the node's LAN address, joins the Gloo group and rebuilds your model, dataset, optimizer and loss from the job spec.
5. **Train.** Gradients are all-reduced directly between the Pis. Rank 0 commits an atomic checkpoint after every epoch.
6. **Monitor.** The controller polls every rank. A job is complete only when **every** rank explicitly reports completed. A failed status query never counts as success.
7. **Recover or finish.** On a crash the whole group is stopped, rescheduled and resumed from the newest checkpoint, up to `max_restarts` times. Errors that no restart can fix (a missing dataset, a bad import path) fail immediately. On success the final weights land in `runs/<job_id>/final_model.pt` and are loaded into your local model.

Nothing sent over the network is pickled or evaluated. Workers receive import paths (`"module:function"`) and JSON settings, and import code that is already installed on the worker.

---

## Installation

You need one controller (Linux, e.g. Ubuntu) and one or more workers (Raspberry Pi 5 with 64-bit Raspberry Pi OS is what Optimus is built and tested on). All machines must share a LAN and run **the same PyTorch version**, which Gloo requires.

### Required system packages

Discovery shells out to two tools. Install them on the controller (workers get them from `install.sh`):

| apt package | provides | used for |
|---|---|---|
| `nmap` | `nmap` | port-scanning the LAN for `optimusd` (port 45870) |
| `avahi-utils` | `avahi-resolve` | resolving IPs to `.local` hostnames |

```bash
sudo apt-get install -y nmap avahi-utils
```

If either is missing Optimus fails gracefully: without `nmap`, `optimus discover` and the dashboard's Scan button stop with an install hint; without `avahi-utils`, names fall back to `getent`/reverse DNS (possibly bare IPs) with a warning.

### 1. Controller

```bash
sudo apt-get update
sudo apt-get install -y nmap avahi-utils nfs-kernel-server python3-pip python3-venv

git clone https://github.com/IsaacAneek/OptimusCluster.git OptimusDL
cd OptimusDL
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Create the data and checkpoint folders and export them to your LAN (use your own subnet in `/etc/exports`):

```bash
sudo mkdir -p /srv/datasets /srv/checkpoints /var/cache/optimus
sudo chown -R $USER:$USER /srv/datasets /srv/checkpoints /var/cache/optimus
```

```text
/srv/datasets    <your-subnet>(ro,sync,no_subtree_check,no_root_squash)
/srv/checkpoints <your-subnet>(rw,sync,no_subtree_check,no_root_squash)
```

```bash
sudo exportfs -ra
sudo systemctl restart nfs-kernel-server
```

### 2. Each worker (repeat per Pi)

Workers run from a git clone that lives on the controller's NFS share, so there is nothing to copy around: `git pull` once and every worker sees it. Mount the share that holds the clone on the Pi (the way you prefer; Optimus does not touch that mount), then:

```bash
cd <path-to-clone>
./deploy/install.sh
```

`deploy/install.sh` is idempotent (safe to re-run after every update). It:

- installs `nmap`, `avahi-utils`, `python3-pip`, `python3-yaml` (and `nfs-common` when NFS is configured) with apt;
- runs `pip3 install -e . --no-deps --break-system-packages`;
- creates the cache, checkpoint and dataset-mount directories named in `config.yaml`;
- adds/mounts the dataset and checkpoint NFS mounts (from `dataset.nfs_server`, `dataset.nfs_export`, `training.checkpoint_nfs_export`) in `/etc/fstab`;
- disables the old `optimus-worker` service if present;
- installs, enables and restarts the `optimusd` systemd unit (from `deploy/optimusd.service`, filled in with the clone path and the current user).

Preview without changing anything with `./deploy/install.sh --dry-run`. For a quick foreground run: `optimusd --config config.yaml`.

### 3. Generate `config.yaml` (host discovery)

There are no hostnames or IPs in the code or in the example config. The worker list is generated by scanning your LAN for `optimusd`:

```bash
optimus discover          # run on the controller, after the workers are up
```

This detects your subnet from the local interface, port-scans it with `nmap` for port 45870, resolves each hostname with `avahi-resolve` (falling back to `getent`, then reverse DNS), checks each host with `GET /v1/status` using the cluster secret, and also ping-sweeps the subnet and records every live host under `hosts:` (ip, hostname, whether it runs optimusd), and writes the `workers:` list (plus `discovery.subnet` and `dataset.nfs_server`) to `config.yaml`. A new `config.yaml` gets a random cluster secret; because the clone is shared, workers read the same file.

```bash
optimus discover --subnet 10.0.0.0/24   # scan a specific subnet
optimus discover --no-verify            # trust the open port, skip /v1/status
optimus discover --merge                # keep workers that were not found this time
optimus discover --dry-run              # print, write nothing
optimus discover --output other.yaml    # write elsewhere
```

The dashboard has the same thing: **Scan for workers** runs the discovery, rewrites `config.yaml` and reloads the worker list. If `config.yaml` is missing or has no workers, the SDK and CLI stop with a message telling you to run discovery. Hosts are never read from anywhere else. `config.yaml` is generated, per-LAN and holds your secret, so it is git-ignored; `example/config.yaml` is a template with an empty `workers:` list.

### 4. Check it works

```bash
optimus discover --dry-run     # lists the workers it can see
curl -H "Authorization: Bearer <secret from config.yaml>" http://<worker>:45870/v1/status
```

A JSON status reply means the daemon is up and the secret matches. Then, from the controller:

```bash
python -c "from optimus import Optimus; print(Optimus(config='config.yaml').probe())"
```

You should see a table with each worker `AVAILABLE`.

> **Important:** the model code you train must be importable by the same module name on the controller **and on every worker**. See [Usage](#usage). With the shared clone, `git pull` is enough.

---

## Usage

### Run the bundled CIFAR-10 example

From the repository root on the controller:

```bash
python example/train_cifar10.py --epochs 3 --workers 2
```

It downloads a CIFAR-10 sample into `/srv/datasets/cifar10/v1`, trains a small ConvNet across your Pis and loads the trained weights back. Other examples:

| Script | What it shows |
|---|---|
| `example/train_cifar10.py` | CIFAR-10 ConvNet with custom model and dataset factories |
| `example/train_resnet.py` | ResNet-18 on an image-folder dataset (`example/example_dataset.py` generates one) |
| `example/raw_ddp_minimal.py` | Bare PyTorch DDP, for comparison |
| `example/evaluate_performance.py` | Cache and scaling benchmark |

### Write your own job

**1. Put your model and dataset code in an importable module** that exists on the controller and every worker (for example `mylab/models.py` in the repo root, which is where `optimusd` runs):

```python
# mylab/models.py
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms

def build_model(num_classes=10):
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 256),
                         nn.ReLU(), nn.Linear(256, num_classes))

def build_dataset(data_dir):            # data_dir = this worker's copy of the data
    tf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    return ImageFolder(data_dir, transform=tf)
```

**2. Put your data** in `/srv/datasets/<name>/<version>/`, as one folder per class for `ImageFolder`.

**3. Write the script:**

```python
import torch, torch.nn as nn
from optimus import Optimus, Dataset, Trainer
from mylab.models import build_model

cluster = Optimus(config="config.yaml")
print(cluster.probe())                               # live worker telemetry

dataset = Dataset(source="/srv/datasets/mydata/v1", dataset_id="mydata", version="v1")

trainer = Trainer(
    cluster=cluster,
    dataset=dataset,
    model_factory="mylab.models:build_model",        # import paths, resolved on every worker
    dataset_factory="mylab.models:build_dataset",
    model_kwargs={"num_classes": 10},
    optimizer=torch.optim.SGD(build_model().parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4),
    loss_fn=nn.CrossEntropyLoss(),
    epochs=5,
    workers=2,
    batch_size=32,
)
metrics = trainer.fit()
print(metrics["final_loss"], metrics["final_model_path"])
```

Optimus builds a local copy of the model from `model_factory` (available as `trainer.model`) and loads the trained weights into it after `fit()`.

### `Trainer` arguments

| Argument | Meaning | Default |
|---|---|---|
| `cluster`, `dataset` | Your `Optimus` instance and `Dataset` | required |
| `model_factory` | `"module:function"` returning an `nn.Module`; `num_classes` is passed if the function accepts it | required |
| `dataset_factory` | `"module:function(data_dir, **dataset_kwargs)"` returning a torch `Dataset` | ImageFolder over the class folders |
| `model_kwargs`, `dataset_kwargs` | JSON-serializable dicts | `{}` |
| `optimizer` | A built-in `torch.optim` optimizer with one parameter group. Workers rebuild it with the same settings (momentum, weight decay, betas, ...) | `SGD(lr)` |
| `loss_fn` | A built-in `torch.nn` loss instance. Workers rebuild it with the same settings | `CrossEntropyLoss()` |
| `epochs`, `batch_size`, `lr` | Training length, per-worker batch size, learning rate | `10`, `32`, `0.01` |
| `workers` | Number of Pis (DDP world size) | `scheduler.default_workers` |
| `allow_synthetic_data` | Let workers train on random tensors when they have no dataset. Off by default | `False` |

Optimizers or losses Optimus cannot recreate on a worker (custom subclasses, function losses, per-group settings, weight tensors) are **rejected with a clear error** instead of being silently replaced.

### What you get back

- `runs/<job_id>/final_model.pt`: final weights.
- `runs/<job_id>/training_metrics.json`, `config.json`, `selected_nodes.json`.
- Checkpoints in `/srv/checkpoints/<job_id>/`.
- Per-rank logs on each Pi in `logs/<job_id>_rank<N>.log`. If a rank fails, the tail of its log is included in the error you see.

---

## The control-plane API

Every worker runs `optimusd`, a passive REST server built on the Python standard library (`http.server`). Plain HTTP/1.1 with JSON bodies, port `45870`. Training traffic (gradients) never goes through it.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/status` | availability, active jobs, uptime |
| `GET` | `/v1/resources` | CPU, RAM, cache disk, SoC temperature |
| `GET` | `/v1/cache/summary`, `/v1/cache/inventory` | cache hits/misses and files with SHA-256 |
| `GET` | `/v1/checkpoints` | saved checkpoints, disk space, mount status |
| `POST` | `/v1/jobs` | submit a job (`201` accepted, `409` rejected) |
| `GET` | `/v1/jobs/{id}` | state, epoch, step, loss, throughput, error |
| `POST` | `/v1/jobs/{id}/stop` | stop a job |
| `POST` | `/v1/shutdown` | stop the daemon |

- **Auth:** every request carries `Authorization: Bearer <cluster.secret>`; the daemon compares it in constant time.
- **Versioning:** `X-Optimus-Protocol-Version: 1` header and `/v1` paths. Unknown JSON fields are ignored, so the protocol can grow.
- **Limits and errors:** bodies over 4 MiB get `413`. Errors are `{"error_code": ..., "error_message": ...}`.

From Python, `optimus.controller.communication.WorkerClient` wraps every endpoint.

---

## Scheduling

One scheduler (`controller/scheduler.py`) serves both the SDK and the dashboard.

**Filter.** A worker is skipped if it is not `AVAILABLE`, already has a job, lacks free RAM, or lacks disk for the dataset files it is missing.

**Score** (each term normalised to 0..1):

```
score = 0.50 * locality + 0.20 * ram + 0.15 * cpu + 0.10 * disk + 0.05 * thermal
```

`locality` is the share of the dataset's bytes already cached on the node. `thermal` falls off as the SoC rises above 35 C. The top *N* workers get ranks `0..N-1`, and the choice is saved to `runs/<job_id>/selected_nodes.json`.

---

## Training, checkpoints and recovery

**Gloo DDP.** Workers use PyTorch `DistributedDataParallel` on the CPU Gloo backend. Optimus finds each node's real LAN interface and address, which avoids the Debian `127.0.1.1` loopback trap that breaks multi-node rendezvous.

**Atomic checkpoints.** After each epoch rank 0 writes `/srv/checkpoints/<job>/checkpoint-<epoch>/` (`model.pt`, `optimizer.pt`, `rng.pt`, `metadata.json`) to a temporary directory, flushes it, and renames it into place. `latest.json` is updated atomically and only the newest 3 checkpoints are kept.

**Fail-stop recovery.**

1. The controller sees a worker go `UNAVAILABLE`, report failure, or stay unreachable for 5 consecutive status checks.
2. It stops every remaining rank in the group.
3. If restarts remain (`training.max_restarts`, default 3), it reschedules on healthy nodes and relaunches with `resume_from_checkpoint`.
4. Workers load weights and optimizer state and continue from epoch *E+1*.

**Non-retryable errors.** A missing dataset, a bad factory path or a failed import stops the job at once, with the worker's traceback, rather than burning every restart.

**No silent fake data.** If a worker cannot find the dataset under `/mnt/nfs-origin/<id>/<version>` or `/srv/datasets/<id>/<version>`, it raises `DatasetUnavailableError` and lists the paths it tried.

---

## Dataset caching

Datasets are immutable, versioned folders on the controller. `Dataset(...)` produces a manifest of every file's path, size and SHA-256. Workers read the data through the NFS mount. A pure-Python cache engine (`cache/manager.py`) is available for staged reads: it takes `fcntl.flock` locks, streams files in 64 KiB chunks, verifies SHA-256 while streaming, and commits atomically into `/var/cache/optimus`. A SQLite database in WAL mode (`cache/registry.py`) tracks what each node holds, and the scheduler uses that for locality scoring.

---

## Dashboard

```bash
uvicorn optimus.dashboard.app:app --host 0.0.0.0 --port 8080
```

Open `http://<controller>:8080`. It is built with Alpine.js on Pico CSS, themed in an IBM Carbon style, and everything is vendored, so it works with no internet. Views:

- **Overview**: worker counts, active jobs, cache size, checkpoint totals.
- **LAN topology**: `nmap` sweep plus `avahi` names to classify gateway, controller, workers and other devices.
- **Workers**: CPU, RAM, disk, thermals, uptime per node.
- **Scheduler**: try a rank allocation for any dataset.
- **Jobs**, **Dataset cache**, **Checkpoints**: including per-node mount checks and a metadata viewer.

The dashboard reads live cluster state through its own `Optimus` instance. Jobs started from another process do not appear in its job list yet, but checkpoints do (they are read from disk).

---

## Configuration

`config.yaml` (the same file on the controller and every worker):

| Key | Meaning |
|---|---|
| `cluster.secret` | Shared bearer token. **Change it.** |
| `controller.tcp_port` | REST port used for workers (default `45870`) |
| `workers[]` | Hostnames (and optional ports) of your Pis |
| `communication.*` | Request timeout, probe interval, failure threshold, max body size |
| `scheduler.default_workers` | Workers used when `Trainer(workers=...)` is omitted |
| `scheduler.output_directory` | Where `runs/<job>/` is written |
| `training.max_restarts` | Fail-stop restart budget |
| `training.checkpoint_directory` | Checkpoint root |
| `logging.directory` | Rank log and status-file location |

Several other keys in the sample file are reserved and not read yet (for example `discovery.scan_interval` and `training.backend`).

---

## Known limitations

Optimus is an MVP prototype. Know these before you rely on it:

- **Code is not shipped.** Your model code must already exist, under the same module name, on the controller and every worker. A model defined inside your `__main__` script is rejected.
- **Shared storage is required.** Datasets and checkpoints move over NFS. Without a shared `/srv/checkpoints`, final weights and resume will not work.
- **Same environment everywhere.** Mismatched PyTorch versions between nodes are not detected.
- **Training is classification-shaped.** The loop expects `(inputs, labels)` batches and supports no LR scheduler, validation pass or custom step yet.
- **Logs stay on the Pis.** Progress is reported per epoch, and `fit()` blocks without a progress display.
- **Security is basic.** A shared secret over plain HTTP, meant for a trusted LAN. Do not expose `optimusd` to the internet.
- **Setup uses root-owned paths** (`/srv/...`) by default.

---

## Repository layout

```
.
|-- config.yaml                    cluster configuration
|-- deploy/                        install.sh (idempotent worker setup) + optimusd.service template
|-- src/optimus/discover.py, cli.py   LAN discovery and the `optimus discover` command
|-- assets/logo.svg
|-- src/optimus/
|   |-- sdk.py  dataset.py  trainer.py  cluster.py     developer-facing SDK
|   |-- protocol/                  REST routes, JSON message dataclasses
|   |-- controller/                probing, discovery, scheduler, launcher,
|   |                              job manager, checkpoints, REST client
|   |-- worker/                    optimusd server, daemon, executor, resource reader
|   |-- cache/                     atomic cache engine and SQLite registry
|   |-- training/                  Gloo setup, sampler, worker entrypoint, factories
|   `-- dashboard/                 FastAPI app, Alpine/Pico UI
|-- example/                       CIFAR-10, ResNet, DDP and benchmark scripts
`-- test/                          pytest suite
```

---

## Development and tests

```bash
pip install -e .
pytest -q
```

The suite covers the REST API, probing, discovery, scheduling, the cache engine and its concurrency, checkpoints, a real Gloo DDP run, worker training with custom factories and optimizers, the job monitor, and the dashboard.

Benchmarks from earlier runs on two Pi 5 nodes (`python example/evaluate_performance.py`): ResNet-18 on Gloo CPU reached 70.9 samples/s on 1 worker and 129.0 samples/s on 2, about 91% scaling efficiency.

<div align="center">

<br/>

**Optimus: small machines, one model.**

</div>
