"""
Optimus Node Discovery Manager.
Integrates Avahi for mDNS hostname resolution, Nmap for subnet scanning,
and protocol verification over TCP.
"""

from dataclasses import dataclass
import logging
import re
import shutil
import socket
import subprocess
from typing import List, Optional

from optimus.cluster import OptimusConfig
from optimus.controller.communication import WorkerClient, WorkerCommunicationError
from optimus.protocol.generated.messages_pb2 import (
    StatusRequest,
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


class NodeDiscoveryManager:
    """Manages discovery and resolution of cluster worker nodes."""

    def __init__(self, config: OptimusConfig):
        self.config = config
        self.avahi_cmd = shutil.which("avahi-resolve-host-name") or shutil.which("avahi-resolve")
        self.nmap_cmd = shutil.which("nmap")

    def resolve_mdns(self, hostname: str, timeout: float = 1.5) -> Optional[str]:
        """
        Resolves an mDNS hostname (e.g. 'pi02.local') using Avahi with fallback
        to standard system getaddrinfo resolver.
        """
        # 1. Try Avahi CLI if installed
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
                    # Output format: <hostname>\t<ip>
                    parts = result.stdout.strip().split()
                    if len(parts) >= 2:
                        ip = parts[1]
                        logger.debug(f"Avahi resolved {hostname} -> {ip}")
                        return ip
            except Exception as e:
                logger.debug(f"Avahi resolution failed for {hostname}: {e}")

        # 2. Fallback to standard resolver (nss-mdns)
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
        """
        Connects to the endpoint over TCP and verifies it responds to the Optimus protocol.
        """
        client = WorkerClient(
            hostname=ip_or_host,
            port=port,
            timeout=timeout,
            cluster_secret=self.config.cluster.secret,
            controller_id=self.config.controller.hostname,
            max_message_size=self.config.communication.max_message_size,
        )
        try:
            req = StatusRequest(probe_only=True)
            resp_env = client.send_request("status_request", req)
            if resp_env.WhichOneof("payload") == "status_response":
                return resp_env.status_response
            return None
        except Exception as e:
            logger.debug(f"Worker verification failed for {ip_or_host}:{port} - {e}")
            return None
        finally:
            client.close()

    def discover(self) -> List[DiscoveredNode]:
        """
        Executes full discovery workflow:
        1. Resolves configured worker hostnames via Avahi/mDNS.
        2. Optionally scans subnet.
        3. Verifies each candidate via TCP Optimus protocol.
        """
        discovered: List[DiscoveredNode] = []

        # Step 1: Configured hostnames
        for worker_cfg in self.config.workers:
            if not worker_cfg.enabled:
                continue

            hostname = worker_cfg.hostname
            port = worker_cfg.port
            ip = self.resolve_mdns(hostname)
            target = ip if ip else hostname

            # Verify node actually speaks Optimus protocol
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
