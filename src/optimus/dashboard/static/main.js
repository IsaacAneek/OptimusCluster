document.addEventListener("DOMContentLoaded", () => {
    fetchOverview();
    fetchWorkers();
    fetchCache();
    fetchJobs();

    // Auto-poll status every 4 seconds
    setInterval(() => {
        fetchOverview();
        fetchWorkers();
        fetchJobs();
    }, 4000);

    // Refresh button
    document.getElementById("btn-refresh").addEventListener("click", async () => {
        const btn = document.getElementById("btn-refresh");
        btn.textContent = "Probing...";
        btn.disabled = true;
        try {
            await fetch("/api/probe", { method: "POST" });
            await fetchOverview();
            await fetchWorkers();
            await fetchCache();
        } catch (e) {
            console.error("Probe failed", e);
        } finally {
            btn.textContent = "Refresh Probes";
            btn.disabled = false;
        }
    });

    // Schedule form submission
    document.getElementById("schedule-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const btn = document.getElementById("btn-schedule");
        btn.textContent = "Scheduling...";
        btn.disabled = true;

        const dataset_id = document.getElementById("sched-dataset").value;
        const dataset_version = document.getElementById("sched-version").value;
        const workers = parseInt(document.getElementById("sched-workers").value, 10);

        try {
            const res = await fetch("/api/schedule", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ dataset_id, dataset_version, workers })
            });

            if (!res.ok) {
                const err = await res.json();
                alert(`Scheduling error: ${err.detail || "Failed to schedule"}`);
                return;
            }

            const data = await res.json();
            renderScheduleResult(data);
        } catch (err) {
            alert(`Scheduling error: ${err.message}`);
        } finally {
            btn.textContent = "Schedule Workers";
            btn.disabled = false;
        }
    });
});

async function fetchOverview() {
    try {
        const res = await fetch("/api/overview");
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById("val-controller").textContent = data.controller_hostname;
        document.getElementById("val-available").textContent = data.available_workers;
        document.getElementById("val-unavailable").textContent = data.unavailable_workers;
        document.getElementById("val-jobs").textContent = data.active_jobs;
        document.getElementById("val-cached").textContent = formatBytes(data.total_cached_bytes);
    } catch (e) {
        console.error("Overview error", e);
    }
}

async function fetchWorkers() {
    try {
        const res = await fetch("/api/workers");
        if (!res.ok) return;
        const workers = await res.json();
        const tbody = document.getElementById("workers-body");

        if (workers.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-muted">No configured workers found</td></tr>';
            return;
        }

        tbody.innerHTML = workers.map(w => {
            const ram = w.ram_total_mb > 0 ? `${(w.ram_available_mb / 1024).toFixed(1)} / ${(w.ram_total_mb / 1024).toFixed(1)} GB` : "N/A";
            const disk = w.cache_disk_available_gb > 0 ? `${w.cache_disk_available_gb} GB` : "N/A";
            const temp = w.temperature_c !== null ? `${w.temperature_c.toFixed(1)} °C` : "N/A";
            return `
                <tr>
                    <td><strong>${w.worker_id}</strong></td>
                    <td>${w.hostname}</td>
                    <td><span class="badge badge-${w.status}">${w.status}</span></td>
                    <td>${w.cpu_percent.toFixed(1)}%</td>
                    <td>${ram}</td>
                    <td>${disk}</td>
                    <td>${temp}</td>
                    <td>${w.active_jobs}</td>
                </tr>
            `;
        }).join("");
    } catch (e) {
        console.error("Workers error", e);
    }
}

async function fetchCache() {
    try {
        const res = await fetch("/api/cache?dataset_id=animals&version=v1");
        if (!res.ok) return;
        const cache = await res.json();
        const tbody = document.getElementById("cache-body");

        if (cache.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-muted">No cache records reported</td></tr>';
            return;
        }

        tbody.innerHTML = cache.map(c => `
            <tr>
                <td>${c.hostname}</td>
                <td>${c.dataset_id}</td>
                <td>${c.version}</td>
                <td>${c.cached_files || 0}</td>
                <td>${formatBytes(c.cached_bytes || 0)}</td>
                <td>${c.cache_hits || 0}</td>
                <td>${c.cache_misses || 0}</td>
                <td>${c.revision || 0}</td>
            </tr>
        `).join("");
    } catch (e) {
        console.error("Cache error", e);
    }
}

async function fetchJobs() {
    try {
        const res = await fetch("/api/jobs");
        if (!res.ok) return;
        const jobs = await res.json();
        const tbody = document.getElementById("jobs-body");

        if (jobs.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="text-muted">No active training jobs recorded</td></tr>';
            return;
        }

        tbody.innerHTML = jobs.map(j => `
            <tr>
                <td><strong>${j.job_id}</strong></td>
                <td>${j.dataset_id}:${j.dataset_version}</td>
                <td><span class="badge badge-${j.status}">${j.status}</span></td>
                <td>${j.epoch}</td>
                <td>${j.global_step}</td>
                <td>${j.loss}</td>
                <td>${j.restarts}</td>
                <td>
                    ${j.status === "JOB_RUNNING" ? `<button class="btn btn-danger-sm" onclick="stopJob('${j.job_id}')">Stop</button>` : "-"}
                </td>
            </tr>
        `).join("");
    } catch (e) {
        console.error("Jobs error", e);
    }
}

async function stopJob(jobId) {
    if (!confirm(`Are you sure you want to stop job ${jobId}?`)) return;
    try {
        await fetch(`/api/jobs/${jobId}/stop`, { method: "POST" });
        await fetchJobs();
    } catch (e) {
        alert("Failed to stop job: " + e.message);
    }
}

function renderScheduleResult(data) {
    const container = document.getElementById("schedule-result");
    document.getElementById("res-job-id").textContent = data.job_id;
    const tbody = document.getElementById("schedule-nodes-body");

    tbody.innerHTML = data.selected_nodes.map(n => `
        <tr>
            <td><strong>Rank ${n.rank}</strong></td>
            <td>${n.hostname}</td>
            <td>${(n.cache_locality * 100).toFixed(1)}%</td>
            <td>${formatBytes(n.cached_bytes)}</td>
            <td>${formatBytes(n.required_bytes)}</td>
            <td><strong>${n.score.toFixed(4)}</strong></td>
        </tr>
    `).join("");

    container.classList.remove("hidden");
}

function formatBytes(bytes) {
    if (!bytes || bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}
