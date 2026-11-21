"""
Tests for LAN host discovery (optimus.discover): nmap/avahi/ip output parsing,
config generation, error handling and the CLI/dashboard entry points.
nmap, avahi and ip are mocked, so these run anywhere.
"""

import socket
import subprocess

import pytest
import yaml
from fastapi.testclient import TestClient

from optimus import discover as d
from optimus.cluster import ConfigError, OptimusConfig
from optimus.sdk import Optimus
from optimus.worker.server import WorkerServer

NMAP_GREPABLE = """\
# Nmap 7.94 scan initiated Sat Nov 21 12:00:00 2026 as: nmap -Pn -n --open -p 45870 -oG - 10.9.8.0/24
Host: 10.9.8.20 ()\tStatus: Up
Host: 10.9.8.20 ()\tPorts: 45870/open/tcp//unknown///\tIgnored State: closed (0)
Host: 10.9.8.3 ()\tPorts: 45870/open/tcp//unknown///
Host: 10.9.8.40 ()\tPorts: 45870/closed/tcp//unknown///
# Nmap done at Sat Nov 21 12:00:03 2026 -- 256 IP addresses (3 hosts up) scanned in 3.00 seconds
"""

IP_ADDR = (
    "1: lo    inet 127.0.0.1/8 scope host lo\\       valid_lft forever\n"
    "2: eth0    inet 10.9.8.5/24 brd 10.9.8.255 scope global eth0\\       valid_lft 100sec\n"
    "3: tailscale0    inet 100.64.0.9/32 scope global tailscale0\\       valid_lft forever\n"
    "4: docker0    inet 172.17.0.1/16 brd 172.17.255.255 scope global docker0\\       valid_lft forever\n"
)


class FakeRun:
    """Maps the first argv element (+ subcommand) to canned subprocess output."""

    def __init__(self, outputs, missing=()):
        self.outputs = outputs
        self.missing = set(missing)
        self.calls = []

    def which(self, name):
        return None if name in self.missing else f"/usr/bin/{name}"

    def run(self, cmd, timeout):
        self.calls.append(cmd)
        out = self.outputs.get(cmd[0], "")
        if isinstance(out, dict):
            out = out.get(cmd[-1], "")
        rc = 0 if out != "" else 1
        return subprocess.CompletedProcess(cmd, rc, stdout=out, stderr="")


@pytest.fixture
def fake(monkeypatch):
    def install(outputs, missing=()):
        f = FakeRun(outputs, missing)
        monkeypatch.setattr(d, "_run", f.run)
        monkeypatch.setattr(d, "_which", f.which)
        return f

    return install


# ---- pure parsers ---------------------------------------------------------
def test_parse_nmap_grepable_only_open_ports_sorted_numerically():
    assert d.parse_nmap_grepable(NMAP_GREPABLE, 45870) == ["10.9.8.3", "10.9.8.20"]
    assert d.parse_nmap_grepable(NMAP_GREPABLE, 22) == []
    assert d.parse_nmap_grepable("", 45870) == []


def test_parse_avahi_and_getent():
    assert d.parse_avahi_address("10.9.8.3\tnode-a.local\n") == "node-a.local"
    assert d.parse_getent_hosts("10.9.8.3       node-a.lan alias\n") == "node-a.lan"
    assert d.parse_avahi_address("") is None


def test_parse_ip_addr_and_default_route():
    assert d.parse_ip_addr(IP_ADDR)[1] == {"iface": "eth0", "cidr": "10.9.8.5/24"}
    assert d.parse_default_route_iface("default via 10.9.8.1 dev eth0 proto dhcp\n") == "eth0"
    assert d.parse_default_route_iface("") is None


# ---- subnet detection -----------------------------------------------------
def test_detect_subnet_prefers_default_route_iface(fake):
    fake({"ip": {"global": IP_ADDR, "default": "default via 10.9.8.1 dev eth0\n"}})
    assert d.detect_local_network() == ("10.9.8.0/24", "10.9.8.5")


def test_detect_subnet_skips_virtual_interfaces_without_default_route(fake):
    fake({"ip": {"global": IP_ADDR.replace("eth0", "wlan0"), "default": ""}})
    assert d.detect_subnet() == "10.9.8.0/24"


def test_detect_subnet_fails_clearly(fake):
    fake({}, missing={"ip"})
    with pytest.raises(d.DiscoveryError, match="--subnet"):
        d.detect_subnet()


# ---- name resolution fallback chain ---------------------------------------
def test_resolve_hostname_prefers_avahi(fake):
    fake({"avahi-resolve": "10.9.8.3\tnode-a.local\n", "getent": "10.9.8.3 other.lan\n"})
    assert d.resolve_hostname("10.9.8.3") == "node-a.local"


def test_resolve_hostname_falls_back_to_getent(fake):
    fake({"getent": "10.9.8.3  node-a.lan\n"}, missing={"avahi-resolve"})
    assert d.resolve_hostname("10.9.8.3") == "node-a.lan"


def test_resolve_hostname_falls_back_to_reverse_dns_then_ip(fake, monkeypatch):
    fake({}, missing={"avahi-resolve", "getent"})
    monkeypatch.setattr(socket, "gethostbyaddr", lambda ip: ("rdns.example", [], [ip]))
    assert d.resolve_hostname("10.9.8.3") == "rdns.example"

    def boom(ip):
        raise socket.herror()

    monkeypatch.setattr(socket, "gethostbyaddr", boom)
    assert d.resolve_hostname("10.9.8.3") == "10.9.8.3"


# ---- run_discovery --------------------------------------------------------
def _outputs():
    return {
        "nmap": NMAP_GREPABLE,
        "avahi-resolve": {"10.9.8.3": "10.9.8.3\tnode-a.local\n", "10.9.8.20": "10.9.8.20\tnode-b.local\n"},
        "ip": {"global": IP_ADDR, "default": "default via 10.9.8.1 dev eth0\n"},
    }


def test_run_discovery_autodetects_subnet_and_resolves(fake):
    f = fake(_outputs())
    res = d.run_discovery(verify=False)
    assert res.subnet == "10.9.8.0/24" and res.local_ip == "10.9.8.5"
    assert [(h.ip, h.hostname) for h in res.hosts] == [("10.9.8.3", "node-a.local"), ("10.9.8.20", "node-b.local")]
    nmap_cmd = next(c for c in f.calls if c[0] == "nmap")
    assert "45870" in nmap_cmd and "10.9.8.0/24" in nmap_cmd


def test_run_discovery_explicit_subnet_and_port(fake):
    f = fake(_outputs())
    res = d.run_discovery(subnet="10.9.8.0/24", port=45999, verify=False)
    nmap_cmd = next(c for c in f.calls if c[0] == "nmap")
    assert "45999" in nmap_cmd
    assert res.hosts == []  # canned output is for 45870


def test_run_discovery_excludes_local_host(fake):
    out = _outputs()
    out["nmap"] += "Host: 10.9.8.5 ()\tPorts: 45870/open/tcp//unknown///\n"
    fake(out)
    assert "10.9.8.5" not in [h.ip for h in d.run_discovery(verify=False).hosts]
    assert "10.9.8.5" in [h.ip for h in d.run_discovery(verify=False, include_local=True).hosts]


def test_run_discovery_without_nmap_errors_with_install_hint(fake):
    fake(_outputs(), missing={"nmap"})
    with pytest.raises(d.DiscoveryError, match="apt-get install"):
        d.run_discovery(verify=False)


def test_run_discovery_without_avahi_warns_but_works(fake):
    out = _outputs()
    out["getent"] = "10.9.8.3 node-a.lan\n"
    fake(out, missing={"avahi-resolve"})
    res = d.run_discovery(verify=False)
    assert any("avahi-utils" in w for w in res.warnings)
    assert res.hosts[0].hostname == "node-a.lan"


def test_run_discovery_verifies_against_real_status_endpoint(fake, monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    worker = WorkerServer(worker_id="w-test", hostname="127.0.0.1", port=port, cluster_secret="s3cret")
    worker.start()
    try:
        fake({"nmap": f"Host: 127.0.0.1 ()\tPorts: {port}/open/tcp//x///\n",
              "avahi-resolve": "127.0.0.1\tlocalnode.local\n"})
        good = d.run_discovery(subnet="127.0.0.0/30", port=port, secret="s3cret", include_local=True)
        assert [(h.hostname, h.verified, h.worker_id) for h in good.hosts] == [("localnode.local", True, "w-test")]
        bad = d.run_discovery(subnet="127.0.0.0/30", port=port, secret="wrong", include_local=True)
        assert bad.hosts == [] and any("/v1/status failed" in w for w in bad.warnings)
        unverified = d.run_discovery(subnet="127.0.0.0/30", port=port, include_local=True)
        assert len(unverified.hosts) == 1 and any("verification" in w for w in unverified.warnings)
    finally:
        worker.stop()


# ---- config generation ----------------------------------------------------
def _result(*names):
    hosts = [d.DiscoveredHost(ip=f"10.9.8.{i + 10}", hostname=n) for i, n in enumerate(names)]
    return d.DiscoveryResult(subnet="10.9.8.0/24", port=45870, hosts=hosts, local_ip="10.9.8.5")


def test_write_config_creates_loadable_file(tmp_path):
    path = tmp_path / "config.yaml"
    d.write_config(str(path), _result("a.local", "b.local"))
    text = path.read_text()
    assert text.startswith("# Generated by `optimus discover`")
    cfg = OptimusConfig.load(str(path))
    assert [w.hostname for w in cfg.workers] == ["a.local", "b.local"]
    assert cfg.discovery.subnet == "10.9.8.0/24"
    assert cfg.cluster.secret  # generated, not a hardcoded default
    assert cfg.dataset.nfs_server == "10.9.8.5"
    assert oct(path.stat().st_mode & 0o777) == "0o640"


def test_write_config_preserves_settings_and_worker_flags(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "cluster": {"secret": "keep-me"},
        "cache": {"max_capacity_gb": 7},
        "workers": [{"hostname": "a.local", "enabled": False}, {"hostname": "gone.local"}],
    }))
    d.write_config(str(path), _result("a.local", "new.local"))
    data = yaml.safe_load(path.read_text())
    assert data["cluster"]["secret"] == "keep-me" and data["cache"]["max_capacity_gb"] == 7
    assert data["workers"] == [{"hostname": "a.local", "enabled": False},
                               {"hostname": "new.local", "enabled": True}]

    d.write_config(str(path), _result("new.local"), merge=True)
    names = [w["hostname"] for w in yaml.safe_load(path.read_text())["workers"]]
    assert names == ["new.local", "a.local"]  # a.local kept from the previous list


def test_discover_and_write_leaves_config_alone_when_nothing_found(fake, tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"cluster": {"secret": "x"}, "workers": [{"hostname": "old.local"}]}))
    out = _outputs()
    out["nmap"] = "# nothing\n"
    fake(out)
    res = d.discover_and_write(str(path), verify=False)
    assert res.hosts == [] and any("unchanged" in w for w in res.warnings)
    assert yaml.safe_load(path.read_text())["workers"] == [{"hostname": "old.local"}]


def test_discover_and_write_output_path_keeps_base_settings(fake, tmp_path):
    base = tmp_path / "config.yaml"
    base.write_text(yaml.safe_dump({"cluster": {"secret": "x"}, "cache": {"max_capacity_gb": 3}}))
    fake(_outputs())
    out = tmp_path / "other.yaml"
    d.discover_and_write(str(base), verify=False, output_path=str(out))
    data = yaml.safe_load(out.read_text())
    assert data["cache"]["max_capacity_gb"] == 3 and len(data["workers"]) == 2
    assert "workers" not in yaml.safe_load(base.read_text())


# ---- code reads hosts only from the generated config ----------------------
def test_missing_config_error_has_discovery_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="optimus discover"):
        Optimus(config=str(tmp_path / "nope.yaml"))


@pytest.mark.parametrize("body", [{"cluster": {"secret": "x"}}, {"workers": []},
                                  {"workers": [{"hostname": "a.local", "enabled": False}]}])
def test_empty_worker_list_error_has_discovery_hint(tmp_path, body):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(body))
    with pytest.raises(ConfigError, match="optimus discover"):
        Optimus(config=str(path))
    Optimus(config=str(path), allow_empty=True).close()  # dashboard mode


# ---- CLI ------------------------------------------------------------------
def test_cli_discover_writes_config(fake, tmp_path, capsys):
    from optimus import cli

    fake(_outputs())
    path = tmp_path / "config.yaml"
    with pytest.raises(SystemExit) as e:
        cli.main(["discover", "--config", str(path), "--no-verify"])
    assert e.value.code == 0
    assert "node-a.local" in capsys.readouterr().out
    assert len(yaml.safe_load(path.read_text())["workers"]) == 2


def test_cli_discover_reports_missing_nmap(fake, tmp_path, capsys):
    from optimus import cli

    fake(_outputs(), missing={"nmap"})
    with pytest.raises(SystemExit) as e:
        cli.main(["discover", "--config", str(tmp_path / "c.yaml"), "--no-verify"])
    assert e.value.code == 1
    assert "nmap is not installed" in capsys.readouterr().err
    assert not (tmp_path / "c.yaml").exists()


# ---- dashboard Scan endpoint ----------------------------------------------
def test_dashboard_scan_updates_config_and_workers(fake, tmp_path, monkeypatch):
    from optimus.dashboard import app as dash

    path = tmp_path / "config.yaml"
    monkeypatch.setenv("OPTIMUS_CONFIG", str(path))
    monkeypatch.setattr(dash, "cluster_instance", None)
    fake(_outputs())
    client = TestClient(dash.app)
    try:
        r = client.post("/api/discovery/scan")
        assert r.status_code == 200
        body = r.json()
        assert body["config_updated"] and [h["hostname"] for h in body["hosts"]] == ["node-a.local", "node-b.local"]
        assert len(yaml.safe_load(path.read_text())["workers"]) == 2
        assert {w["hostname"] for w in client.get("/api/workers").json()} == {"node-a.local", "node-b.local"}
    finally:
        dash.cluster_instance.close()
        monkeypatch.setattr(dash, "cluster_instance", None)


def test_dashboard_scan_reports_missing_nmap(fake, tmp_path, monkeypatch):
    from optimus.dashboard import app as dash

    monkeypatch.setenv("OPTIMUS_CONFIG", str(tmp_path / "config.yaml"))
    monkeypatch.setattr(dash, "cluster_instance", None)
    fake(_outputs(), missing={"nmap"})
    r = TestClient(dash.app).post("/api/discovery/scan")
    assert r.status_code == 503 and "nmap is not installed" in r.json()["detail"]
    dash.cluster_instance.close()
    monkeypatch.setattr(dash, "cluster_instance", None)


def test_parse_nmap_up_hosts():
    assert d.parse_nmap_up_hosts(NMAP_GREPABLE) == ["10.9.8.20"]


def test_config_lists_every_live_lan_host(fake, tmp_path):
    out = _outputs()
    out["nmap"] = NMAP_GREPABLE + "Host: 10.9.8.1 ()\tStatus: Up\n"
    fake(out)
    path = tmp_path / "config.yaml"
    d.discover_and_write(str(path), verify=False)
    hosts = yaml.safe_load(path.read_text())["hosts"]
    assert {h["ip"] for h in hosts} == {"10.9.8.1", "10.9.8.3", "10.9.8.20"}
    assert next(h for h in hosts if h["ip"] == "10.9.8.3")["optimusd"] is True
    assert next(h for h in hosts if h["ip"] == "10.9.8.1")["optimusd"] is False
