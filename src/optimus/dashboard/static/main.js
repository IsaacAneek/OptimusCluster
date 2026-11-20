/**
 * Optimus Cluster Dashboard Client-Side Logic.
 * Handles tab navigation (Claude.ai style), live telemetry polling,
 * LAN topology discovery, scheduling, and job management.
 */

document.addEventListener("DOMContentLoaded", () => {
    // 1. Navigation Controller (Claude.ai sidebar navigation)
    const navItems = document.querySelectorAll(".nav-item");
    const tabViews = document.querySelectorAll(".tab-view");
    const pageTitle = document.getElementById("page-title");

    const viewTitles = {
        "view-overview": "Overview",
        "view-discovery": "LAN Topology Discovery",
        "view-workers": "Cluster Workers",
        "view-scheduler": "Locality Scheduler",
        "view-jobs": "Distributed Training Jobs",
        "view-cache": "Dataset Cache",
        "view-checkpoints": "Saved Node Checkpoints",
    };

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const targetView = item.getAttribute("data-view");

            navItems.forEach(i => i.classList.remove("active"));
            item.classList.add("active");

            tabViews.forEach(v => v.classList.remove("active"));
            const activeView = document.getElementById(targetView);
            if (activeView) activeView.classList.add("active");

            if (pageTitle && viewTitles[targetView]) {
                pageTitle.textContent = viewTitles[targetView];
            }

            // Trigger view-specific loads
            if (targetView === "view-discovery") {
                fetchDiscovery();
            } else if (targetView === "view-jobs") {
                fetchJobs();
            } else if (targetView === "view-cache") {
                fetchCache();
            } else if (targetView === "view-checkpoints") {
                fetchCheckpoints();
            }
        });
    });

    // 2. Fetch Overview Metrics
    async function fetchOverview() {
        try {
            const res = await fetch("/api/overview");
            if (!res.ok) return;
            const data = await res.json();

            document.getElementById("val-available").textContent = data.available_workers;
            document.getElementById("val-busy").textContent = data.busy_workers;
            document.getElementById("val-unavailable").textContent = data.unavailable_workers;
            document.getElementById("val-jobs").textContent = data.active_jobs;

            const mb = (data.total_cached_bytes / (1024 * 1024)).toFixed(1);
            document.getElementById("val-cached").textContent = `${mb} MB`;

            const ckptVal = document.getElementById("val-checkpoints");
            if (ckptVal && data.total_checkpoints !== undefined) {
                ckptVal.textContent = data.total_checkpoints;
            }

            document.getElementById("sidebar-controller").textContent = data.controller_hostname || "isaac.local";
            document.getElementById("sidebar-subnet").textContent = data.subnet || "192.168.10.0/24";

            const pill = document.getElementById("cluster-status-pill");
            if (pill) {
                if (data.available_workers > 0) {
                    pill.className = "pill pill-primary";
                    pill.textContent = `${data.available_workers} Node${data.available_workers > 1 ? 's' : ''} Online`;
                } else {
                    pill.className = "pill pill-outline-dark";
                    pill.textContent = "No Workers Online";
                }
            }
        } catch (e) {
            console.error("Failed to fetch overview:", e);
        }
    }

    // 3. Fetch Workers Telemetry
    async function fetchWorkers() {
        try {
            const res = await fetch("/api/workers");
            if (!res.ok) return;
            const workers = await res.json();

            renderWorkersTable("overview-workers-body", workers, false);
            renderWorkersTable("workers-body", workers, true);
        } catch (e) {
            console.error("Failed to fetch workers:", e);
        }
    }

    function renderWorkersTable(elementId, workers, isDetailed) {
        const tbody = document.getElementById(elementId);
        if (!tbody) return;

        if (workers.length === 0) {
            tbody.innerHTML = `<tr><td colspan="${isDetailed ? 9 : 8}" class="text-muted">No worker nodes registered</td></tr>`;
            return;
        }

        tbody.innerHTML = workers.map(w => {
            const statusClass = w.status === "AVAILABLE" ? "pill pill-primary" : "pill pill-outline-dark";
            const tempStr = w.temperature_c !== null ? `${w.temperature_c.toFixed(1)}°C` : "N/A";
            const cpuStr = `${w.cpu_percent.toFixed(1)}%`;
            const ramStr = `${w.ram_available_mb} / ${w.ram_total_mb} MB`;
            const diskStr = `${w.cache_disk_available_gb} GB`;

            if (isDetailed) {
                return `
                    <tr>
                        <td><strong>${w.worker_id}</strong></td>
                        <td>${w.hostname}</td>
                        <td>${w.port}</td>
                        <td><span class="${statusClass}">${w.status}</span></td>
                        <td>${cpuStr}</td>
                        <td>${ramStr}</td>
                        <td>${diskStr}</td>
                        <td>${tempStr}</td>
                        <td>${w.consecutive_failures}</td>
                    </tr>
                `;
            } else {
                return `
                    <tr>
                        <td><strong>${w.worker_id}</strong></td>
                        <td>${w.hostname}</td>
                        <td><span class="${statusClass}">${w.status}</span></td>
                        <td>${cpuStr}</td>
                        <td>${ramStr}</td>
                        <td>${diskStr}</td>
                        <td>${tempStr}</td>
                        <td>${w.active_jobs}</td>
                    </tr>
                `;
            }
        }).join("");
    }

    // 4. Fetch LAN Topology Discovery (Nmap & Avahi)
    async function fetchDiscovery() {
        const tbody = document.getElementById("discovery-body");
        if (!tbody) return;

        try {
            const res = await fetch("/api/discovery");
            if (!res.ok) return;
            const hosts = await res.json();
            renderDiscoveryTable(hosts);
        } catch (e) {
            console.error("Failed to fetch discovery:", e);
        }
    }

    function renderDiscoveryTable(hosts) {
        const tbody = document.getElementById("discovery-body");
        if (!tbody) return;

        if (hosts.length === 0) {
            tbody.innerHTML = `<tr><td colspan="6" class="text-muted">No LAN scan results yet. Click "Run Subnet Sweep" to scan.</td></tr>`;
            return;
        }

        tbody.innerHTML = hosts.map(h => {
            let entityType = "LAN Host";
            if (h.is_gateway) entityType = "Gateway / Router";
            else if (h.is_controller) entityType = "Optimus Controller";
            else if (h.is_optimus_worker) entityType = "Optimus Worker Node";

            const isWorkerPill = h.is_optimus_worker 
                ? `<span class="pill pill-primary">YES</span>`
                : `<span class="pill pill-outline">No</span>`;

            const statusPill = h.status
                ? `<span class="${h.status === 'AVAILABLE' ? 'pill pill-primary' : 'pill pill-outline-dark'}">${h.status}</span>`
                : `<span class="text-muted">-</span>`;

            return `
                <tr>
                    <td><strong>${h.ip}</strong></td>
                    <td>${h.hostname}</td>
                    <td>${entityType}</td>
                    <td>${isWorkerPill}</td>
                    <td>${h.worker_id || "-"}</td>
                    <td>${statusPill}</td>
                </tr>
            `;
        }).join("");
    }

    // Trigger LAN Subnet Sweep (Nmap + Avahi)
    async function triggerScanLan() {
        const btn1 = document.getElementById("btn-scan-lan");
        const btn2 = document.getElementById("btn-trigger-scan");
        const oldText1 = btn1 ? btn1.textContent : "";
        const oldText2 = btn2 ? btn2.textContent : "";

        if (btn1) { btn1.textContent = "Scanning..."; btn1.disabled = true; }
        if (btn2) { btn2.textContent = "Scanning Subnet..."; btn2.disabled = true; }

        try {
            const res = await fetch("/api/discovery/scan", { method: "POST" });
            if (res.ok) {
                const hosts = await res.json();
                renderDiscoveryTable(hosts);
            }
        } catch (e) {
            console.error("LAN scan failed:", e);
        } finally {
            if (btn1) { btn1.textContent = oldText1; btn1.disabled = false; }
            if (btn2) { btn2.textContent = oldText2; btn2.disabled = false; }
        }
    }

    document.getElementById("btn-scan-lan")?.addEventListener("click", triggerScanLan);
    document.getElementById("btn-trigger-scan")?.addEventListener("click", triggerScanLan);

    // 5. Trigger Probes Refresh
    document.getElementById("btn-refresh")?.addEventListener("click", async () => {
        const btn = document.getElementById("btn-refresh");
        const orig = btn.textContent;
        btn.textContent = "Probing...";
        btn.disabled = true;
        try {
            await fetch("/api/probe", { method: "POST" });
            await fetchOverview();
            await fetchWorkers();
        } catch (e) {
            console.error("Probing failed:", e);
        } finally {
            btn.textContent = orig;
            btn.disabled = false;
        }
    });

    // 6. Fetch Jobs
    async function fetchJobs() {
        const tbody = document.getElementById("jobs-body");
        if (!tbody) return;

        try {
            const res = await fetch("/api/jobs");
            if (!res.ok) return;
            const jobs = await res.json();

            if (jobs.length === 0) {
                tbody.innerHTML = `<tr><td colspan="9" class="text-muted">No training jobs recorded</td></tr>`;
                return;
            }

            tbody.innerHTML = jobs.map(j => {
                const statusClass = j.status === "JOB_RUNNING" ? "pill pill-primary" : "pill pill-outline-dark";
                const isRunning = j.status === "JOB_RUNNING";
                const actionBtn = isRunning
                    ? `<button class="btn btn-secondary btn-sm" onclick="stopJob('${j.job_id}')">Stop</button>`
                    : `<span class="text-muted">-</span>`;

                return `
                    <tr>
                        <td><strong>${j.job_id}</strong></td>
                        <td>${j.dataset_id}:${j.dataset_version}</td>
                        <td><span class="${statusClass}">${j.status}</span></td>
                        <td>${j.world_size}</td>
                        <td>${j.epoch}</td>
                        <td>${j.global_step}</td>
                        <td>${j.loss}</td>
                        <td>${j.restarts}</td>
                        <td>${actionBtn}</td>
                    </tr>
                `;
            }).join("");
        } catch (e) {
            console.error("Failed to fetch jobs:", e);
        }
    }

    window.stopJob = async function(jobId) {
        if (!confirm(`Stop job ${jobId}?`)) return;
        try {
            await fetch(`/api/jobs/${jobId}/stop`, { method: "POST" });
            fetchJobs();
            fetchOverview();
        } catch (e) {
            console.error("Failed to stop job:", e);
        }
    };

    // 7. Fetch Cache Inventory
    async function fetchCache() {
        const tbody = document.getElementById("cache-body");
        if (!tbody) return;

        try {
            const res = await fetch("/api/cache");
            if (!res.ok) return;
            const data = await res.json();

            if (data.length === 0) {
                tbody.innerHTML = `<tr><td colspan="9" class="text-muted">No cache records found</td></tr>`;
                return;
            }

            tbody.innerHTML = data.map(c => {
                if (c.error) {
                    return `
                        <tr>
                            <td><strong>${c.hostname}</strong></td>
                            <td>${c.dataset_id}</td>
                            <td>${c.version}</td>
                            <td colspan="6" class="text-muted">Error: ${c.error}</td>
                        </tr>
                    `;
                }
                const mb = (c.cached_bytes / (1024 * 1024)).toFixed(2);
                const dlMb = (c.downloaded_bytes / (1024 * 1024)).toFixed(2);
                return `
                    <tr>
                        <td><strong>${c.hostname}</strong></td>
                        <td>${c.dataset_id}</td>
                        <td>${c.version}</td>
                        <td>${c.cached_files}</td>
                        <td>${mb} MB</td>
                        <td><strong>${c.cache_hits}</strong></td>
                        <td>${c.cache_misses}</td>
                        <td>${dlMb} MB</td>
                        <td>rev.${c.revision}</td>
                    </tr>
                `;
            }).join("");
        } catch (e) {
            console.error("Failed to fetch cache:", e);
        }
    }

    // 8. Scheduler Form Handling
    const scheduleForm = document.getElementById("schedule-form");
    if (scheduleForm) {
        scheduleForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const btn = document.getElementById("btn-schedule");
            btn.disabled = true;
            btn.textContent = "Computing Locality...";

            const payload = {
                dataset_id: document.getElementById("sched-dataset").value,
                dataset_version: document.getElementById("sched-version").value,
                workers: parseInt(document.getElementById("sched-workers").value, 10),
            };

            try {
                const res = await fetch("/api/schedule", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });

                if (!res.ok) {
                    const err = await res.json();
                    alert(`Scheduling failed: ${err.detail || 'Unknown error'}`);
                    return;
                }

                const result = await res.json();
                document.getElementById("res-job-id").textContent = result.job_id;
                const ranksBody = document.getElementById("schedule-ranks-body");
                ranksBody.innerHTML = result.ranks.map(r => `
                    <tr>
                        <td><strong>Rank ${r.rank}</strong></td>
                        <td>${r.hostname}</td>
                        <td>${r.worker_id}</td>
                        <td><strong>${r.score.toFixed(3)}</strong></td>
                        <td>${(r.locality * 100).toFixed(1)}%</td>
                        <td>${(r.cached_bytes / (1024 * 1024)).toFixed(1)} MB</td>
                    </tr>
                `).join("");

                document.getElementById("schedule-result").classList.remove("hidden");
            } catch (err) {
                alert(`Error scheduling: ${err.message}`);
            } finally {
                btn.disabled = false;
                btn.textContent = "Schedule Ranks";
            }
        });
    }

    // 8. Fetch and Manage Checkpoints
    let cachedCheckpointsData = null;
    let currentFilterNode = "";

    async function fetchCheckpoints(nodeFilter = "") {
        currentFilterNode = nodeFilter;
        const tbody = document.getElementById("checkpoints-body");
        if (!tbody) return;

        try {
            const res = await fetch("/api/checkpoints");
            if (!res.ok) return;
            const data = await res.json();
            cachedCheckpointsData = data;

            // Render summary metrics
            if (data.summary) {
                document.getElementById("ckpt-total-count").textContent = data.summary.total_checkpoints || 0;
                document.getElementById("ckpt-total-size").textContent = `${data.summary.total_size_mb || 0} MB`;
                document.getElementById("ckpt-nodes-accessible").textContent = `${data.summary.nodes_accessible_count} / ${data.summary.nodes_count} Mounted`;
                document.getElementById("ckpt-latest-info").textContent =
                    (data.summary.latest_job && data.summary.latest_checkpoint)
                        ? `${data.summary.latest_job} (${data.summary.latest_checkpoint})`
                        : "-";
            }

            // Render node verification cards
            renderNodeCards(data.nodes || []);

            // Render filter buttons
            renderFilterBar(data.nodes || []);

            // Render checkpoints table
            renderCheckpointsTable(data.checkpoints || [], currentFilterNode);
        } catch (e) {
            console.error("Failed to fetch checkpoints:", e);
            tbody.innerHTML = `<tr><td colspan="10" class="text-muted">Error loading checkpoints: ${e.message}</td></tr>`;
        }
    }

    function renderNodeCards(nodes) {
        const container = document.getElementById("node-checkpoints-grid");
        if (!container) return;

        container.innerHTML = nodes.map(n => `
            <div class="node-checkpoint-card ${n.accessible ? 'card-important' : ''}">
                <div class="node-card-header">
                    <div>
                        <div class="node-card-title">${n.hostname}</div>
                        <div class="node-card-meta">${n.role} • ${n.worker_id}</div>
                    </div>
                    <span class="pill ${n.accessible ? 'pill-primary' : 'pill-outline-dark'}">
                        ${n.accessible ? 'VERIFIED' : 'UNAVAILABLE'}
                    </span>
                </div>
                <div class="node-card-meta font-mono" style="word-break: break-all;">
                    ${n.checkpoint_dir}
                </div>
                <div class="node-card-stats">
                    <div>
                        <div class="card-label" style="margin-bottom: 2px;">Checkpoints</div>
                        <strong>${n.checkpoints_count}</strong>
                    </div>
                    <div>
                        <div class="card-label" style="margin-bottom: 2px;">Disk Free</div>
                        <strong>${n.disk_available_gb} GB</strong>
                    </div>
                    <div>
                        <div class="card-label" style="margin-bottom: 2px;">Status</div>
                        <strong>${n.status}</strong>
                    </div>
                </div>
            </div>
        `).join("");
    }

    function renderFilterBar(nodes) {
        const bar = document.getElementById("ckpt-filter-bar");
        if (!bar) return;

        const buttons = [
            `<button class="filter-btn ${currentFilterNode === "" ? 'active' : ''}" data-node="">All Nodes</button>`
        ];

        nodes.forEach(n => {
            const isActive = currentFilterNode === n.hostname ? "active" : "";
            buttons.push(`<button class="filter-btn ${isActive}" data-node="${n.hostname}">${n.hostname}</button>`);
        });

        bar.innerHTML = buttons.join("");

        bar.querySelectorAll(".filter-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const selected = btn.getAttribute("data-node");
                currentFilterNode = selected;
                bar.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                if (cachedCheckpointsData) {
                    renderCheckpointsTable(cachedCheckpointsData.checkpoints || [], currentFilterNode);
                }
            });
        });
    }

    function renderCheckpointsTable(checkpoints, filterNode) {
        const tbody = document.getElementById("checkpoints-body");
        if (!tbody) return;

        let filtered = checkpoints;
        if (filterNode) {
            filtered = checkpoints.filter(c => {
                return (c.verified_nodes && c.verified_nodes.includes(filterNode)) ||
                       (c.saved_by_node && c.saved_by_node.includes(filterNode));
            });
        }

        if (filtered.length === 0) {
            tbody.innerHTML = `<tr><td colspan="10" class="text-muted">${filterNode ? `No checkpoints found matching node ${filterNode}` : 'No saved checkpoints found in cluster'}</td></tr>`;
            return;
        }

        tbody.innerHTML = filtered.map(c => {
            const verifiedBadges = (c.verified_nodes || []).map(nodeHost =>
                `<span class="pill pill-primary" style="font-size: 0.72rem; padding: 2px 6px; margin: 1px;">${nodeHost}</span>`
            ).join(" ");

            const latestBadge = c.is_latest
                ? `<span class="pill pill-primary" style="font-size: 0.72rem; padding: 2px 6px; margin-left: 4px;">LATEST</span>`
                : "";

            return `
                <tr>
                    <td class="font-mono"><strong>${c.job_id}</strong></td>
                    <td><span class="font-mono">${c.checkpoint_name}</span> ${latestBadge}</td>
                    <td>${c.epoch}</td>
                    <td>${c.global_step}</td>
                    <td>${c.loss.toFixed(4)}</td>
                    <td><span class="font-mono">${c.saved_by_node}</span></td>
                    <td>${verifiedBadges || '<span class="text-muted">Unverified</span>'}</td>
                    <td>${c.model_size_mb} MB</td>
                    <td><span class="text-muted">${c.created_at}</span></td>
                    <td>
                        <button class="btn btn-secondary btn-sm btn-inspect-ckpt" data-job="${c.job_id}" data-ckpt="${c.checkpoint_name}">
                            Inspect
                        </button>
                    </td>
                </tr>
            `;
        }).join("");

        // Bind Inspect buttons
        tbody.querySelectorAll(".btn-inspect-ckpt").forEach(btn => {
            btn.addEventListener("click", async () => {
                const jobId = btn.getAttribute("data-job");
                const ckptName = btn.getAttribute("data-ckpt");
                await inspectCheckpoint(jobId, ckptName);
            });
        });
    }

    async function inspectCheckpoint(jobId, ckptName) {
        const card = document.getElementById("checkpoint-detail-card");
        if (!card) return;

        try {
            const res = await fetch(`/api/checkpoints/${jobId}/${ckptName}/metadata`);
            if (!res.ok) {
                alert(`Failed to load metadata for ${ckptName}`);
                return;
            }
            const meta = await res.json();

            const item = ((cachedCheckpointsData && cachedCheckpointsData.checkpoints) || []).find(
                c => c.job_id === jobId && c.checkpoint_name === ckptName
            );

            document.getElementById("detail-ckpt-name").textContent = `${jobId} / ${ckptName}`;
            document.getElementById("detail-ckpt-path").textContent = item ? item.path : `/srv/checkpoints/${jobId}/${ckptName}`;
            document.getElementById("detail-ckpt-dataset").textContent = item ? `${item.dataset_id}:${item.dataset_version}` : "-";
            document.getElementById("detail-ckpt-files").textContent = item ? item.saved_files.join(", ") : "-";
            document.getElementById("detail-ckpt-json").textContent = JSON.stringify(meta, null, 2);

            card.classList.remove("hidden");
            card.scrollIntoView({ behavior: "smooth", block: "nearest" });
        } catch (err) {
            alert(`Error: ${err.message}`);
        }
    }

    // Detail card close button
    const btnCloseDetail = document.getElementById("btn-close-detail");
    if (btnCloseDetail) {
        btnCloseDetail.addEventListener("click", () => {
            const card = document.getElementById("checkpoint-detail-card");
            if (card) card.classList.add("hidden");
        });
    }

    // Refresh checkpoints button
    const btnRefreshCkpts = document.getElementById("btn-refresh-checkpoints");
    if (btnRefreshCkpts) {
        btnRefreshCkpts.addEventListener("click", () => {
            fetchCheckpoints(currentFilterNode);
        });
    }

    // Initial load & Polling Loop
    fetchOverview();
    fetchWorkers();
    setInterval(fetchOverview, 4000);
    setInterval(fetchWorkers, 4000);
});
