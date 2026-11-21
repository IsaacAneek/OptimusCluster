"""
Optimus Node Discovery Manager.
Integrates Avahi for mDNS hostname resolution, Nmap for subnet scanning,
and protocol verification over TCP.
"""

from dataclasses import asdict, dataclass
import logging
import re
import shutil
import socket
import subprocess
from typing import Any, Dict, List, Optional

from optimus.cluster import OptimusConfig
from optimus.controller.communication import WorkerClient, WorkerCommunicationError
from optimus.protocol.messages import (
    StatusResponse,
    WorkerState,
)

logger = logging.getLogger("optimus.controller.discovery")


@dataclass
class DiscoveredNode:
    worker_id: str
    hostname: str
    ip_address: str
    port: int
    state: WorkerState
    active_jobs: int
    uptime_seconds: int


@dataclass
class LanHostInfo:
    ip: str
    hostname: str
    is_optimus_worker: bool
    worker_id: Optional[str] = None
    status: Optional[str] = None
    port: Optional[int] = None
    is_controller: bool = False
    is_gateway: bool = False
    active_jobs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class NodeDiscoveryManager:
    """Manages discovery and resolution of cluster worker nodes and LAN topology."""

    def __init__(self, config: OptimusConfig):
        self.config = config
        self.avahi_cmd = shutil.which("avahi-resolve-host-name") or shutil.which("avahi-resolve")
        self.avahi_addr_cmd = shutil.which("avahi-resolve-address") or shutil.which("avahi-resolve")
        self.nmap_cmd = shutil.which("nmap")
        self._cached_lan_hosts: List[LanHostInfo] = []

    def resolve_mdns(self, hostname: str, timeout: float = 1.5) -> Optional[str]:
        """
        Resolves an mDNS hostname (e.g. 'pi02.local') using Avahi with fallback
        to standard system getaddrinfo resolver.
        """
        if self.avahi_cmd:
            try:
                cmd = [self.avahi_cmd]
                if "avahi-resolve" in self.avahi_cmd and "-host-name" not in self.avahi_cmd:
                    cmd.extend(["-n", hostname])
                else:
                    cmd.append(hostname)

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                if result.returncode == 0 and result.stdout:
                    parts = result.stdout.strip().split()
                    if len(parts) >= 2:
                        ip = parts[1]
                        logger.debug(f"Avahi resolved {hostname} -> {ip}")
                        return ip
            except Exception as e:
                logger.debug(f"Avahi resolution failed for {hostname}: {e}")

        try:
            addr_info = socket.getaddrinfo(
                hostname, None, socket.AF_INET, socket.SOCK_STREAM
            )
            if addr_info:
                ip = addr_info[0][4][0]
                logger.debug(f"Socket resolved {hostname} -> {ip}")
                return ip
        except Exception as e:
            logger.debug(f"System DNS resolution failed for {hostname}: {e}")

        return None

    def resolve_ip_to_hostname(self, ip: str, timeout: float = 1.5) -> str:
        """Resolves an IP to mDNS or DNS hostname using Avahi / socket."""
        if self.avahi_addr_cmd:
            try:
                cmd = [self.avahi_addr_cmd]
                if "avahi-resolve" in self.avahi_addr_cmd and "-address" not in self.avahi_addr_cmd:
                    cmd.extend(["-a", ip])
                else:
                    cmd.append(ip)

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                if result.returncode == 0 and result.stdout:
                    parts = result.stdout.strip().split()
                    if len(parts) >= 2:
                        return parts[1]
            except Exception:
                pass

        try:
            host, _, _ = socket.gethostbyaddr(ip)
            return host
        except Exception:
            return ip

    def scan_subnet(self, subnet: str, timeout: float = 15.0) -> List[str]:
        """
        Scans a LAN subnet for live hosts using Nmap ping sweep.
        Returns list of live IP addresses.
        """
        if not self.nmap_cmd:
            logger.warning("Nmap command not found in PATH; skipping subnet scan")
            return []

        try:
            cmd = [self.nmap_cmd, "-sn", subnet, "-oG", "-"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            live_ips = []
            for line in result.stdout.splitlines():
                if "Status: Up" in line:
                    match = re.search(r"Host:\s+([\d\.]+)", line)
                    if match:
                        live_ips.append(match.group(1))
            logger.info(f"Nmap discovered {len(live_ips)} live hosts on {subnet}")
            return live_ips
        except Exception as e:
            logger.error(f"Nmap scan failed on {subnet}: {e}")
            return []

    def verify_worker(
        self,
        ip_or_host: str,
        port: int,
        timeout: float = 2.0,
    ) -> Optional[StatusResponse]:
        """Calls GET /v1/status on the endpoint to verify it is an optimusd worker."""
        client = WorkerClient(
            hostname=ip_or_host,
            port=port,
            timeout=timeout,
            cluster_secret=self.config.cluster.secret,
            controller_id=self.config.controller.hostname,
            max_message_size=self.config.communication.max_message_size,
        )
        try:
            status = client.get_status()
            return status if status.worker_id else None
        except Exception as e:
            logger.debug(f"Worker verification failed for {ip_or_host}:{port} - {e}")
            return None
        finally:
            client.close()

    def scan_lan_topology(self, subnet: Optional[str] = None) -> List[LanHostInfo]:
        """
        Discovers all live hosts on the LAN via Nmap, resolves hostnames via Avahi,
        and identifies Optimus workers vs other network entities.
        """
        target_subnet = subnet or self.config.discovery.subnet or "192.168.10.0/24"
        live_ips = self.scan_subnet(target_subnet)
        default_port = self.config.controller.tcp_port or 45870

        # Local controller IP
        try:
            controller_ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            controller_ip = "127.0.0.1"

        hosts: List[LanHostInfo] = []

        for ip in live_ips:
            hostname = self.resolve_ip_to_hostname(ip)
            is_gw = ip.endswith(".1")
            is_ctrl = (ip == controller_ip) or (hostname == self.config.controller.hostname)

            # Test if it's an Optimus worker
            worker_status = None
            if not is_gw:
                worker_status = self.verify_worker(ip, default_port, timeout=1.0)
                if not worker_status and hostname != ip:
                    worker_status = self.verify_worker(hostname, default_port, timeout=1.0)

            if worker_status:
                hosts.append(
                    LanHostInfo(
                        ip=ip,
                        hostname=hostname,
                        is_optimus_worker=True,
                        worker_id=worker_status.worker_id,
                        status=WorkerState.Name(worker_status.status),
                        port=default_port,
                        is_controller=is_ctrl,
                        is_gateway=is_gw,
                        active_jobs=worker_status.active_job_count,
                    )
                )
            else:
                hosts.append(
                    LanHostInfo(
                        ip=ip,
                        hostname=hostname,
                        is_optimus_worker=False,
                        is_controller=is_ctrl,
                        is_gateway=is_gw,
                    )
                )

        self._cached_lan_hosts = hosts
        return hosts

    def get_cached_lan_hosts(self) -> List[LanHostInfo]:
        return self._cached_lan_hosts

    def discover(self) -> List[DiscoveredNode]:
        """Executes worker discovery for configured hosts."""
        discovered: List[DiscoveredNode] = []

        for worker_cfg in self.config.workers:
            if not worker_cfg.enabled:
                continue

            hostname = worker_cfg.hostname
            port = worker_cfg.port
            ip = self.resolve_mdns(hostname)
            target = ip if ip else hostname

            status = self.verify_worker(target, port)
            if status:
                discovered.append(
                    DiscoveredNode(
                        worker_id=status.worker_id,
                        hostname=hostname,
                        ip_address=ip or target,
                        port=port,
                        state=status.status,
                        active_jobs=status.active_job_count,
                        uptime_seconds=status.uptime_seconds,
                    )
                )
                logger.info(
                    f"Discovered and verified worker '{status.worker_id}' at {hostname} ({target}:{port})"
                )
            else:
                logger.debug(f"Worker {hostname} ({target}:{port}) is currently unreachable")

        return discovered
