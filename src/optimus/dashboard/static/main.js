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

    // Initial load & Polling Loop
    fetchOverview();
    fetchWorkers();
    setInterval(fetchOverview, 4000);
    setInterval(fetchWorkers, 4000);
});
