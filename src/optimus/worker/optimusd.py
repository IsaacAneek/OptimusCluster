"""
optimusd: the Optimus worker daemon.
Lightweight background process running on each worker node. Serves the
control-plane REST API for the controller and executes training workloads.

Run with ``optimusd --config config.yaml`` or ``python -m optimus.worker.optimusd``.
"""

import argparse
import logging
import os
import signal
import sys
import time

from optimus.cluster import OptimusConfig
from optimus.worker.server import WorkerServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("optimusd")


class OptimusDaemon:
    def __init__(
        self,
        worker_id: str,
        hostname: str,
        port: int = 45870,
        cluster_secret: str = "",
        config: OptimusConfig = None,
    ):
        self.worker_id = worker_id
        self.hostname = hostname
        self.port = port
        self.cluster_secret = cluster_secret
        self.config = config

        self.server = WorkerServer(
            worker_id=self.worker_id,
            hostname=self.hostname,
            port=self.port,
            cluster_secret=self.cluster_secret,
            max_message_size=(
                config.communication.max_message_size if config else 4194304
            ),
            cache_dir=(
                config.cache.directory if config else "/var/cache/optimus"
            ),
        )
        self.server.on_shutdown = self._request_exit
        self._running = False

    def _request_exit(self) -> None:
        logger.info("Shutdown requested via REST API, exiting...")
        self._running = False

    def start(self) -> None:
        logger.info(f"Starting optimusd '{self.worker_id}' on {self.hostname}:{self.port}")
        self.server.start()
        self._running = True

        def _sig_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)

        while self._running:
            time.sleep(1.0)
        self.stop()

    def stop(self) -> None:
        self._running = False
        self.server.stop()
        logger.info("optimusd stopped successfully.")


# Backwards-compatible alias
WorkerDaemon = OptimusDaemon


def main():
    parser = argparse.ArgumentParser(prog="optimusd", description="optimusd: Optimus worker daemon (REST control-plane API)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--worker-id", type=str, default=None, help="Worker ID (defaults to hostname)")
    parser.add_argument("--hostname", type=str, default=None, help="Worker hostname")
    parser.add_argument("--port", type=int, default=None, help="HTTP port for the REST API")
    args = parser.parse_args()

    config = None
    if os.path.exists(args.config):
        config = OptimusConfig.load(args.config)

    import socket
    local_hostname = args.hostname or socket.gethostname()
    worker_id = args.worker_id or f"worker-{local_hostname}"
    port = args.port or (config.controller.tcp_port if config else 45870)
    secret = config.cluster.secret if config else ""

    daemon = OptimusDaemon(
        worker_id=worker_id,
        hostname=local_hostname,
        port=port,
        cluster_secret=secret,
        config=config,
    )
    daemon.start()


if __name__ == "__main__":
    main()
