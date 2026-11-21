"""
Optimus host discovery.

Scans the LAN for optimusd workers and generates the ``workers`` section of
``config.yaml``. Nothing in the code base hardcodes hostnames or addresses;
the config file written here is the only source of truth for the host list.

Pipeline:
  1. subnet      - detected from the local interface unless given explicitly
  2. nmap        - port scan for the optimusd port (default 45870)
  3. avahi       - ``avahi-resolve -a`` for names, falling back to
                   ``getent hosts`` and then reverse DNS
  4. /v1/status  - optional check that the open port really is optimusd
  5. config.yaml - written atomically, preserving unrelated settings
"""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import secrets
import shutil
import socket
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger("optimus.discover")

DEFAULT_PORT = 45870
INSTALL_HINT = "Install them with: sudo apt-get install -y nmap avahi-utils"
DISCOVER_HINT = "Run `optimus discover` (or press Scan in the dashboard) to generate it."

# Interfaces that never carry the cluster LAN.
_SKIP_IFACE_PREFIXES = ("lo", "docker", "br-", "veth", "virbr", "tailscale", "zt", "wg")


class DiscoveryError(RuntimeError):
    """Raised when discovery cannot run (for example nmap is not installed)."""


@dataclass
class DiscoveredHost:
    ip: str
    hostname: str
    port: int = DEFAULT_PORT
    verified: bool = False
    worker_id: Optional[str] = None
    status: Optional[str] = None
    active_jobs: int = 0
    is_local: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiscoveryResult:
    subnet: str
    port: int
    hosts: List[DiscoveredHost] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    local_ip: Optional[str] = None  # this machine's address on the scanned subnet
    lan_hosts: List[Dict[str, Any]] = field(default_factory=list)  # every live host

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subnet": self.subnet,
            "port": self.port,
            "hosts": [h.to_dict() for h in self.hosts],
            "warnings": list(self.warnings),
            "lan_hosts": list(self.lan_hosts),
        }


# --------------------------------------------------------------------------
# Command runner (single seam so tests can mock nmap / avahi / ip output)
# --------------------------------------------------------------------------
def _run(cmd: List[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _which(name: str) -> Optional[str]:
    return shutil.which(name)


# --------------------------------------------------------------------------
# Parsers (pure functions)
# --------------------------------------------------------------------------
def parse_ip_addr(text: str) -> List[Dict[str, str]]:
    """Parses ``ip -4 -o addr show scope global`` into [{iface, cidr}]."""
    out = []
    for line in text.splitlines():
        m = re.match(r"\s*\d+:\s+(\S+)\s+inet\s+(\d+\.\d+\.\d+\.\d+/\d+)", line)
        if m:
            out.append({"iface": m.group(1), "cidr": m.group(2)})
    return out


def parse_default_route_iface(text: str) -> Optional[str]:
    """Returns the device of the first ``default via ... dev X`` route."""
    m = re.search(r"^default\b.*?\bdev\s+(\S+)", text, re.MULTILINE)
    return m.group(1) if m else None


def parse_nmap_grepable(text: str, port: int) -> List[str]:
    """Returns IPs reported with ``port`` open in nmap ``-oG`` output."""
    ips = []
    needle = f"{port}/open/"
    for line in text.splitlines():
        if not line.startswith("Host:") or "Ports:" not in line:
            continue
        m = re.match(r"Host:\s+(\d+\.\d+\.\d+\.\d+)", line)
        if m and needle in line:
            ips.append(m.group(1))
    return sorted(set(ips), key=lambda s: tuple(int(p) for p in s.split(".")))


def parse_nmap_up_hosts(text: str) -> List[str]:
    """Returns IPs reported ``Status: Up`` in nmap ``-sn -oG`` output."""
    ips = re.findall(r"^Host:\s+(\d+\.\d+\.\d+\.\d+)\s.*Status:\s+Up", text, re.MULTILINE)
    return sorted(set(ips), key=lambda s: tuple(int(p) for p in s.split(".")))


def parse_avahi_address(text: str) -> Optional[str]:
    """``avahi-resolve -a`` prints ``<ip>\\t<name>``; returns the name."""
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            return parts[1].rstrip(".")
    return None


def parse_getent_hosts(text: str) -> Optional[str]:
    """``getent hosts <ip>`` prints ``<ip>  <name> [aliases]``; returns the name."""
    return parse_avahi_address(text)


# --------------------------------------------------------------------------
# Network helpers
# --------------------------------------------------------------------------
def _local_interfaces() -> List[Dict[str, str]]:
    if not _which("ip"):
        return []
    try:
        res = _run(["ip", "-4", "-o", "addr", "show", "scope", "global"], 5)
    except Exception:
        return []
    return parse_ip_addr(res.stdout) if res.returncode == 0 else []


def _default_iface() -> Optional[str]:
    if not _which("ip"):
        return None
    try:
        res = _run(["ip", "-4", "route", "show", "default"], 5)
    except Exception:
        return None
    return parse_default_route_iface(res.stdout) if res.returncode == 0 else None


def detect_local_network() -> "tuple[str, str]":
    """
    Returns (subnet_cidr, local_ip) for the LAN interface: the default-route
    interface if there is one, otherwise the first non-virtual interface.
    """
    ifaces = _local_interfaces()
    wanted = _default_iface()
    candidates = [i for i in ifaces if i["iface"] == wanted] if wanted else []
    candidates += [i for i in ifaces if not i["iface"].startswith(_SKIP_IFACE_PREFIXES)]
    for cand in candidates:
        iface = ipaddress.ip_interface(cand["cidr"])
        if iface.network.prefixlen < 32:
            return str(iface.network), str(iface.ip)
    raise DiscoveryError(
        "Could not detect the local subnet (no usable interface found). "
        "Pass it explicitly, e.g. `optimus discover --subnet 192.168.1.0/24`."
    )


def detect_subnet() -> str:
    return detect_local_network()[0]


def _local_ip_in(subnet: str) -> Optional[str]:
    try:
        net = ipaddress.ip_network(subnet, strict=False)
    except ValueError:
        return None
    for i in _local_interfaces():
        ip = ipaddress.ip_interface(i["cidr"]).ip
        if ip in net:
            return str(ip)
    return None


def resolve_hostname(ip: str, timeout: float = 2.0) -> str:
    """avahi-resolve -a, then ``getent hosts``, then reverse DNS, then the IP."""
    if _which("avahi-resolve"):
        try:
            res = _run(["avahi-resolve", "-a", ip], timeout)
            if res.returncode == 0:
                name = parse_avahi_address(res.stdout)
                if name:
                    return name
        except Exception as e:
            logger.debug("avahi-resolve failed for %s: %s", ip, e)
    if _which("getent"):
        try:
            res = _run(["getent", "hosts", ip], timeout)
            if res.returncode == 0:
                name = parse_getent_hosts(res.stdout)
                if name:
                    return name
        except Exception as e:
            logger.debug("getent failed for %s: %s", ip, e)
    try:
        return socket.gethostbyaddr(ip)[0]
    except Exception:
        return ip


def scan_port(subnet: str, port: int, timeout: float = 120.0) -> List[str]:
    """Port-scans ``subnet`` with nmap and returns IPs with ``port`` open."""
    if not _which("nmap"):
        raise DiscoveryError(f"nmap is not installed. {INSTALL_HINT}")
    cmd = [
        "nmap", "-Pn", "-n", "--open", "-T4", "--max-retries", "1",
        "-p", str(port), "-oG", "-", subnet,
    ]
    try:
        res = _run(cmd, timeout)
    except subprocess.TimeoutExpired:
        raise DiscoveryError(f"nmap scan of {subnet} timed out after {timeout:.0f}s.")
    if res.returncode != 0:
        raise DiscoveryError(f"nmap failed on {subnet}: {res.stderr.strip() or 'exit ' + str(res.returncode)}")
    return parse_nmap_grepable(res.stdout, port)


def scan_live_hosts(subnet: str, timeout: float = 120.0) -> List[str]:
    """Ping-sweeps ``subnet`` with nmap and returns every live IP."""
    if not _which("nmap"):
        raise DiscoveryError(f"nmap is not installed. {INSTALL_HINT}")
    res = _run(["nmap", "-sn", "-n", "-oG", "-", subnet], timeout)
    if res.returncode != 0:
        raise DiscoveryError(f"nmap sweep failed on {subnet}: {res.stderr.strip()}")
    return parse_nmap_up_hosts(res.stdout)


def _verify(host: str, port: int, secret: str, timeout: float):
    from optimus.controller.communication import WorkerClient

    client = WorkerClient(hostname=host, port=port, timeout=timeout, cluster_secret=secret, controller_id="discover")
    try:
        status = client.get_status()
        return status if status.worker_id else None
    except Exception as e:
        logger.debug("verify failed for %s:%s: %s", host, port, e)
        return None
    finally:
        client.close()


def run_discovery(
    subnet: Optional[str] = None,
    port: int = DEFAULT_PORT,
    secret: str = "",
    verify: bool = True,
    include_local: bool = False,
    verify_timeout: float = 2.0,
) -> DiscoveryResult:
    """Finds optimusd hosts. Raises DiscoveryError if nmap is unavailable."""
    local_ip = None
    if not subnet:
        subnet, local_ip = detect_local_network()
    else:
        local_ip = _local_ip_in(subnet)

    result = DiscoveryResult(subnet=subnet, port=port, local_ip=local_ip)
    if not _which("avahi-resolve"):
        result.warnings.append(
            "avahi-utils not installed; hostnames come from getent/reverse DNS and may be bare IPs. "
            + INSTALL_HINT
        )

    do_verify = verify and bool(secret)
    if verify and not secret:
        result.warnings.append("No cluster secret known; skipped /v1/status verification.")

    local_ips = {i["cidr"].split("/")[0] for i in _local_interfaces()}
    for ip in scan_port(subnet, port):
        is_local = ip in local_ips
        if is_local and not include_local:
            continue
        host = DiscoveredHost(ip=ip, hostname=resolve_hostname(ip), port=port, is_local=is_local)
        if do_verify:
            status = _verify(ip, port, secret, verify_timeout)
            if status is None:
                result.warnings.append(
                    f"{host.hostname} ({ip}) has port {port} open but /v1/status failed "
                    "(wrong secret or not optimusd); skipped."
                )
                continue
            from optimus.protocol.messages import WorkerState

            host.verified = True
            host.worker_id = status.worker_id
            host.status = WorkerState.Name(status.status)
            host.active_jobs = status.active_job_count
        result.hosts.append(host)

    try:
        worker_ips = {h.ip for h in result.hosts}
        names = {h.ip: h.hostname for h in result.hosts}
        for ip in sorted(set(scan_live_hosts(subnet)) | worker_ips,
                         key=lambda x: tuple(int(p) for p in x.split("."))):
            result.lan_hosts.append({
                "ip": ip,
                "hostname": names.get(ip) or resolve_hostname(ip),
                "optimusd": ip in worker_ips,
                "local": ip in local_ips,
            })
    except DiscoveryError as e:
        result.warnings.append(f"LAN host sweep failed: {e}")
    return result


# --------------------------------------------------------------------------
# config.yaml generation
# --------------------------------------------------------------------------
_HEADER = (
    "# Generated by `optimus discover` - do not hand-maintain the worker list.\n"
    "# Re-run `optimus discover` (or use Scan in the dashboard) after the cluster changes.\n"
)


def build_config_dict(
    existing: Optional[Dict[str, Any]],
    result: DiscoveryResult,
    secret: Optional[str] = None,
    merge: bool = False,
) -> Dict[str, Any]:
    """Returns the config mapping with the workers/subnet from ``result``."""
    from optimus.cluster import OptimusConfig

    if existing:
        data = dict(existing)
    else:
        base = asdict(OptimusConfig())
        base.pop("workers", None)
        base["controller"]["hostname"] = socket.gethostname()
        base["cluster"]["secret"] = secret or secrets.token_urlsafe(24)
        data = base

    if secret and not data.get("cluster", {}).get("secret"):
        data.setdefault("cluster", {})["secret"] = secret

    disc = data.setdefault("discovery", {})
    disc["subnet"] = result.subnet

    dataset = data.setdefault("dataset", {})
    if result.local_ip and not dataset.get("nfs_server"):
        dataset["nfs_server"] = result.local_ip

    previous = {w["hostname"]: w for w in (data.get("workers") or []) if isinstance(w, dict) and "hostname" in w}
    workers: List[Dict[str, Any]] = []
    seen = set()
    for h in result.hosts:
        entry = dict(previous.get(h.hostname, {}))
        entry["hostname"] = h.hostname
        entry.setdefault("enabled", True)
        if h.port != DEFAULT_PORT:
            entry["port"] = h.port
        workers.append(entry)
        seen.add(h.hostname)
    if merge:
        workers += [w for name, w in previous.items() if name not in seen]
    data["workers"] = workers
    if result.lan_hosts:
        data["hosts"] = list(result.lan_hosts)  # every live LAN host (informational)
    return data


def write_config(
    path: str,
    result: DiscoveryResult,
    secret: Optional[str] = None,
    merge: bool = False,
    base_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Writes ``path`` atomically and returns the data written. Settings are
    carried over from ``base_path`` (default: ``path``) when it exists.
    """
    existing = None
    base = base_path or path
    if os.path.isfile(base):
        with open(base, "r", encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
    data = build_config_dict(existing, result, secret=secret, merge=merge)
    text = _HEADER + yaml.safe_dump(data, sort_keys=False, default_flow_style=False)

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".config-", suffix=".yaml", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        if os.path.exists(path):
            os.chmod(tmp, os.stat(path).st_mode & 0o777)
        else:
            os.chmod(tmp, 0o640)  # contains the cluster secret
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return data


def discover_and_write(
    config_path: str,
    subnet: Optional[str] = None,
    port: Optional[int] = None,
    secret: Optional[str] = None,
    verify: bool = True,
    include_local: bool = False,
    merge: bool = False,
    output_path: Optional[str] = None,
) -> DiscoveryResult:
    """
    Shared entry point for the CLI and the dashboard. Reads defaults (secret,
    subnet, port) from an existing ``config_path``, runs discovery, and writes
    ``output_path`` (default: ``config_path``) only if hosts were found.
    """
    existing: Dict[str, Any] = {}
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
    eff_secret = secret or (existing.get("cluster") or {}).get("secret") or ""
    eff_subnet = subnet or (existing.get("discovery") or {}).get("subnet") or None
    eff_port = port or (existing.get("controller") or {}).get("tcp_port") or DEFAULT_PORT

    result = run_discovery(
        subnet=eff_subnet, port=int(eff_port), secret=eff_secret,
        verify=verify, include_local=include_local,
    )
    target = output_path or config_path
    if result.hosts:
        write_config(target, result, secret=secret, merge=merge, base_path=config_path)
    else:
        result.warnings.append(f"No optimusd hosts found on {result.subnet}:{result.port}; {target} left unchanged.")
    return result

