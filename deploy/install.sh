#!/usr/bin/env bash
# Idempotent worker setup for a Raspberry Pi running from an NFS-hosted git clone.
#
#   cd <clone on NFS> && ./deploy/install.sh [--dry-run] [--config PATH]
#
# Safe to re-run: it installs the package, packages, directories, NFS mounts and
# the optimusd systemd unit, and restarts the service. Everything host-specific
# (directories, NFS server and exports) is read from the generated config.yaml.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO_DIR/config.yaml"
DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --config) CONFIG="$2"; shift ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

SERVICE_USER="${SUDO_USER:-$(id -un)}"
UNIT_DEST=/etc/systemd/system/optimusd.service
PYTHON="$(command -v python3)"

say() { printf '==> %s\n' "$*"; }
run() { if [ "$DRY_RUN" = 1 ]; then printf '[dry-run] %s\n' "$*"; else "$@"; fi; }
priv() { if [ "$(id -u)" = 0 ]; then run "$@"; else run sudo "$@"; fi; }
as_user() { if [ "$(id -un)" = "$SERVICE_USER" ]; then run "$@"; else run sudo -u "$SERVICE_USER" "$@"; fi; }

# --- 1. apt packages ---------------------------------------------------------
PKGS=(nmap avahi-utils python3-pip python3-yaml)
missing=()
for p in "${PKGS[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p"); done

# --- config values (from the generated config.yaml, never hardcoded) --------
if [ ! -f "$CONFIG" ]; then
  echo "warning: $CONFIG not found. Generate it on the controller with 'optimus discover'" >&2
  echo "         (config.yaml lives in the shared clone, so workers pick it up automatically)." >&2
fi
# Python's yaml may not be installed yet on a fresh Pi; install packages first.
if [ "${#missing[@]}" -gt 0 ]; then
  say "Installing apt packages: ${missing[*]}"
  priv apt-get update -qq
  priv apt-get install -y "${missing[@]}"
fi

cfg_values() {
  [ -f "$CONFIG" ] || return 0
  python3 - "$CONFIG" <<'PY'
import shlex, sys, yaml
c = yaml.safe_load(open(sys.argv[1])) or {}
g = lambda sec, key, default="": (c.get(sec) or {}).get(key, default) or default
vals = {
    "CACHE_DIR": g("cache", "directory"),
    "CKPT_DIR": g("training", "checkpoint_directory"),
    "CKPT_EXPORT": g("training", "checkpoint_nfs_export", g("training", "checkpoint_directory")),
    "DATA_MOUNT": g("dataset", "worker_nfs_mount"),
    "DATA_EXPORT": g("dataset", "nfs_export"),
    "NFS_SERVER": g("dataset", "nfs_server"),
}
for k, v in vals.items():
    print(f"{k}={shlex.quote(str(v))}")
PY
}
CACHE_DIR= CKPT_DIR= CKPT_EXPORT= DATA_MOUNT= DATA_EXPORT= NFS_SERVER=
eval "$(cfg_values)"

# --- 2. python package -------------------------------------------------------
say "Installing optimus (editable, no deps)"
( cd "$REPO_DIR" && as_user pip3 install -e . --no-deps --break-system-packages )

# --- 3. directories ----------------------------------------------------------
for d in "$CACHE_DIR" "$CKPT_DIR" "$DATA_MOUNT"; do
  [ -n "$d" ] || continue
  [ -d "$d" ] || { say "Creating $d"; priv mkdir -p "$d"; }
done
for d in "$CACHE_DIR"; do
  [ -n "$d" ] && priv chown "$SERVICE_USER" "$d"
done

# --- 4. NFS mounts (only when the config names an NFS server) ----------------
ensure_mount() {  # export mountpoint options
  local export="$1" mp="$2" opts="$3"
  [ -n "$mp" ] && [ -n "$export" ] || return 0
  local entry="$NFS_SERVER:$export $mp nfs $opts,_netdev,nofail 0 0"
  if ! grep -qsF "$NFS_SERVER:$export $mp " /etc/fstab; then
    say "Adding fstab entry for $mp"
    if [ "$DRY_RUN" = 1 ]; then echo "[dry-run] append to /etc/fstab: $entry"
    else echo "$entry" | priv tee -a /etc/fstab >/dev/null; fi
  fi
  if ! mountpoint -q "$mp" 2>/dev/null; then
    say "Mounting $mp"
    priv mount "$mp" || echo "warning: could not mount $mp (is the NFS server up?)" >&2
  fi
}
if [ -n "$NFS_SERVER" ]; then
  dpkg -s nfs-common >/dev/null 2>&1 || priv apt-get install -y nfs-common
  ensure_mount "$DATA_EXPORT" "$DATA_MOUNT" "ro,soft,timeo=30"
  ensure_mount "$CKPT_EXPORT" "$CKPT_DIR" "rw,soft,timeo=30"
else
  echo "note: dataset.nfs_server is empty in $CONFIG; skipping NFS mounts." >&2
fi

# --- 5. systemd --------------------------------------------------------------
if systemctl list-unit-files optimus-worker.service >/dev/null 2>&1 \
   && systemctl list-unit-files 2>/dev/null | grep -q '^optimus-worker\.service'; then
  say "Disabling old optimus-worker service"
  priv systemctl disable --now optimus-worker.service || true
fi

say "Installing optimusd unit"
unit="$(sed -e "s|@REPO_DIR@|$REPO_DIR|g" -e "s|@SERVICE_USER@|$SERVICE_USER|g" \
            -e "s|@PYTHON@|$PYTHON|g" "$REPO_DIR/deploy/optimusd.service")"
if [ "$DRY_RUN" = 1 ]; then
  echo "[dry-run] write $UNIT_DEST:"; echo "$unit" | sed 's/^/    /'
else
  echo "$unit" | priv tee "$UNIT_DEST" >/dev/null
fi
priv systemctl daemon-reload
priv systemctl enable optimusd.service
priv systemctl restart optimusd.service
say "Done. Check: systemctl status optimusd"
