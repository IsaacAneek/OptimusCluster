#!/usr/bin/env python3
"""
Optimus Cluster CLI.
Validates cluster configuration, discovers nodes, and probes worker status.
Usage:
    python cluster.py --config config.yaml [--watch]
"""

import argparse
import logging
import sys
import time

from optimus.cluster import OptimusConfig
from optimus.controller.communication import WorkerClientManager
from optimus.controller.probing import ProbeManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("optimus.cluster")


def main():
    parser = argparse.ArgumentParser(description="Optimus Cluster Controller CLI")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously probe and monitor worker availability",
    )
    args = parser.parse_args()

    try:
        config = OptimusConfig.load(args.config)
        config.require_workers()
        logger.info(f"Loaded and validated configuration from '{args.config}' (Cluster: {config.cluster.name})")
    except Exception as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    client_mgr = WorkerClientManager(
        default_timeout=config.communication.request_timeout,
        cluster_secret=config.cluster.secret,
        controller_id=config.controller.hostname,
        max_message_size=config.communication.max_message_size,
    )

    probe_mgr = ProbeManager(
        client_manager=client_mgr,
        failure_threshold=config.communication.failure_threshold,
        probe_interval=config.communication.probe_interval,
    )

    # Register configured worker nodes
    for worker in config.workers:
        if worker.enabled:
            probe_mgr.register_worker(
                hostname=worker.hostname,
                port=worker.port,
                worker_id=worker.worker_id,
            )

    logger.info(f"Probing {len(config.workers)} configured workers...")
    probe_mgr.probe_all()
    probe_mgr.probe_all_resources()
    print("\n--- Worker Availability Table ---")
    print(probe_mgr.format_worker_table())
    print("---------------------------------\n")

    if args.watch:
        logger.info(f"Entering continuous watch mode (interval: {config.communication.probe_interval}s). Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(config.communication.probe_interval)
                probe_mgr.probe_all()
                probe_mgr.probe_all_resources()
                print("\033[H\033[J", end="")  # Clear screen
                print(f"--- Worker Availability Table (Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}) ---")
                print(probe_mgr.format_worker_table())
                print("---------------------------------------------------------------------------------")
        except KeyboardInterrupt:
            logger.info("Exiting watch mode.")
        finally:
            client_mgr.close_all()


if __name__ == "__main__":
    main()
