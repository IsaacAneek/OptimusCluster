/*
 * Optimus dashboard state, as a single Alpine.js component.
 * The markup in templates/index.html binds to this declaratively (x-for, x-text,
 * @click); there is no manual DOM manipulation here.
 */
document.addEventListener("alpine:init", () => {
  const MB = 1024 * 1024;

  Alpine.data("dashboard", () => ({
    // --- navigation ---------------------------------------------------------
    views: [
      { id: "overview", label: "Overview", title: "Overview" },
      { id: "discovery", label: "LAN Discovery", title: "LAN Topology Discovery" },
      { id: "workers", label: "Workers", title: "Cluster Workers" },
      { id: "scheduler", label: "Scheduler", title: "Locality Scheduler" },
      { id: "jobs", label: "Training Jobs", title: "Distributed Training Jobs" },
      { id: "cache", label: "Dataset Cache", title: "Dataset Cache" },
      { id: "checkpoints", label: "Checkpoints", title: "Saved Node Checkpoints" },
    ],
    view: "overview",

    // --- data ---------------------------------------------------------------
    overview: {},
    workers: [],
    hosts: [],
    scanInfo: null,
    jobs: [],
    cache: [],
    ckpt: { summary: {}, nodes: [], checkpoints: [] },
    ckptNode: "",
    detail: null,
    schedule: { dataset_id: "animals", dataset_version: "v1", workers: 2 },
    scheduleResult: null,
    busy: { probe: false, scan: false, schedule: false, ckpt: false, inspect: false },

    // --- lifecycle ----------------------------------------------------------
    init() {
      this.refreshLive();
      setInterval(() => this.refreshLive(), 4000);
    },
    get title() {
      return (this.views.find((v) => v.id === this.view) || {}).title;
    },
    go(id) {
      this.view = id;
      const loaders = {
        discovery: () => this.loadHosts(),
        jobs: () => this.loadJobs(),
        cache: () => this.loadCache(),
        checkpoints: () => this.loadCheckpoints(),
      };
      (loaders[id] || (() => {}))();
    },
    refreshLive() {
      this.loadOverview();
      this.loadWorkers();
    },

    // --- helpers ------------------------------------------------------------
    async api(path, opts) {
      const res = await fetch(path, opts);
      if (!res.ok) {
        let detail = res.statusText;
        try { detail = (await res.json()).detail || detail; } catch (_) {}
        throw new Error(detail);
      }
      return res.json();
    },
    async run(key, fn) {
      this.busy[key] = true;
      try { return await fn(); }
      catch (e) { console.error(e); alert(e.message); }
      finally { this.busy[key] = false; }
    },
    mb(bytes, digits = 1) { return ((bytes || 0) / MB).toFixed(digits) + " MB"; },
    healthy(status) { return status === "AVAILABLE" || status === "JOB_RUNNING"; },
    temp(w) { return w.temperature_c == null ? "N/A" : w.temperature_c.toFixed(1) + " °C"; },
    entity(h) {
      if (h.is_gateway) return "Gateway / Router";
      if (h.is_controller) return "Optimus Controller";
      if (h.is_optimus_worker) return "Optimus Worker (optimusd)";
      return "LAN Host";
    },
    get online() { return (this.overview.available_workers || 0) + (this.overview.busy_workers || 0); },
    get filteredCheckpoints() {
      const n = this.ckptNode;
      return (this.ckpt.checkpoints || []).filter(
        (c) => !n || (c.verified_nodes || []).includes(n) || (c.saved_by_node || "").includes(n)
      );
    },

    // --- loaders ------------------------------------------------------------
    async loadOverview() { try { this.overview = await this.api("/api/overview"); } catch (e) { console.error(e); } },
    async loadWorkers() { try { this.workers = await this.api("/api/workers"); } catch (e) { console.error(e); } },
    async loadHosts() { try { this.hosts = await this.api("/api/discovery"); } catch (e) { console.error(e); } },
    async loadJobs() { try { this.jobs = await this.api("/api/jobs"); } catch (e) { console.error(e); } },
    async loadCache() { try { this.cache = await this.api("/api/cache"); } catch (e) { console.error(e); } },
    async loadCheckpoints() {
      await this.run("ckpt", async () => { this.ckpt = await this.api("/api/checkpoints"); });
    },

    // --- actions ------------------------------------------------------------
    probe() {
      return this.run("probe", async () => {
        await this.api("/api/probe", { method: "POST" });
        this.refreshLive();
      });
    },
    scan() {
      return this.run("scan", async () => {
        const r = await this.api("/api/discovery/scan", { method: "POST" });
        this.hosts = r.hosts;
        this.scanInfo = r;
        this.refreshLive();
      });
    },
    submitSchedule() {
      return this.run("schedule", async () => {
        this.scheduleResult = await this.api("/api/schedule", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...this.schedule, workers: Number(this.schedule.workers) }),
        });
      });
    },
    stopJob(id) {
      if (!confirm(`Stop job ${id}?`)) return;
      return this.run("stop-" + id, async () => {
        await this.api(`/api/jobs/${encodeURIComponent(id)}/stop`, { method: "POST" });
        this.loadJobs();
        this.loadOverview();
      });
    },
    inspect(c) {
      return this.run("inspect", async () => {
        const meta = await this.api(
          `/api/checkpoints/${encodeURIComponent(c.job_id)}/${encodeURIComponent(c.checkpoint_name)}/metadata`
        );
        this.detail = { c, json: JSON.stringify(meta, null, 2) };
      });
    },
  }));
});
