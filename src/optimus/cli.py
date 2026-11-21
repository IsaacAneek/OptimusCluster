"""
``optimus`` command line interface.

    optimus discover [--subnet CIDR] [--config config.yaml] ...

Scans the LAN for optimusd workers and generates config.yaml.
"""

import argparse
import logging
import os
import sys

import yaml

from optimus.discover import (
    DEFAULT_PORT,
    DiscoveryError,
    build_config_dict,
    discover_and_write,
    run_discovery,
)


def _cmd_discover(args: argparse.Namespace) -> int:
    try:
        if args.dry_run:
            cfg = {}
            if os.path.isfile(args.config):
                with open(args.config, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
            result = run_discovery(
                subnet=args.subnet or (cfg.get("discovery") or {}).get("subnet") or None,
                port=args.port or (cfg.get("controller") or {}).get("tcp_port") or DEFAULT_PORT,
                secret=args.secret or (cfg.get("cluster") or {}).get("secret") or "",
                verify=not args.no_verify, include_local=args.include_local,
            )
        else:
            result = discover_and_write(
                args.config, subnet=args.subnet, port=args.port, secret=args.secret,
                verify=not args.no_verify, include_local=args.include_local,
                merge=args.merge, output_path=args.output,
            )
    except DiscoveryError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    print(f"Scanned {result.subnet} for optimusd on port {result.port}")
    for h in result.hosts:
        extra = f"  [{h.status}, {h.worker_id}]" if h.verified else ""
        print(f"  {h.ip:<16} {h.hostname}{extra}")
    for w in result.warnings:
        print(f"warning: {w}", file=sys.stderr)

    if not result.hosts:
        return 2
    target = args.output or args.config
    if args.dry_run:
        print("\n(dry run) config that would be written:\n")
        print(yaml.safe_dump(build_config_dict(None, result, secret=args.secret), sort_keys=False))
    else:
        print(f"Wrote {len(result.hosts)} worker(s) to {target}")
    return 0


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="optimus", description="Optimus cluster tools")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("discover", help="scan the LAN for optimusd workers and write config.yaml")
    d.add_argument("--config", default="config.yaml", help="config file to read defaults from and update")
    d.add_argument("--output", default=None, help="write here instead of --config")
    d.add_argument("--subnet", default=None, help="CIDR to scan (default: detected from the local interface)")
    d.add_argument("--port", type=int, default=None, help=f"optimusd port (default: {DEFAULT_PORT})")
    d.add_argument("--secret", default=None, help="cluster secret for /v1/status verification (default: from config)")
    d.add_argument("--no-verify", action="store_true", help="skip the /v1/status check; trust the open port")
    d.add_argument("--include-local", action="store_true", help="also list this machine if it runs optimusd")
    d.add_argument("--merge", action="store_true", help="keep previously configured workers that were not found")
    d.add_argument("--dry-run", action="store_true", help="scan and print; do not write any file")
    d.set_defaults(func=_cmd_discover)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
