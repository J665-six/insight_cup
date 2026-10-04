#!/usr/bin/env bash
set -Eeuo pipefail

if (( EUID != 0 )); then
  printf 'Run as root: sudo %s\n' "$0" >&2
  exit 2
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
NX_SERVER_CONFIG="${NX_SERVER_CONFIG:-/usr/NX/etc/server.cfg}"
NX_NODE_CONFIG="${NX_NODE_CONFIG:-/usr/NX/etc/node.cfg}"
TIME_UNIT="${RK3588_TIME_UNIT:-/etc/systemd/system/rk3588-time-sync.service}"
GDM_TIME_DROPIN="${RK3588_GDM_TIME_DROPIN:-/etc/systemd/system/gdm.service.d/10-rk3588-time-sync.conf}"
NX_DROPIN_DIR="${NX_DROPIN_DIR:-/etc/systemd/system/nxserver.service.d}"
NX_DROPIN="$NX_DROPIN_DIR/10-physical-xorg.conf"
WAIT_HELPER="${NX_WAIT_HELPER:-/usr/local/sbin/nx-wait-physical-xorg}"
GUARD_HELPER="${NX_GUARD_HELPER:-/usr/local/sbin/nx-guard-physical-xorg}"
GUARD_UNIT="${NX_GUARD_UNIT:-/etc/systemd/system/nx-physical-xorg-guard.service}"
RESTART_MARKER="${NX_POLICY_RESTART_MARKER:-/run/nomachine-physical-policy-restart-required}"
timestamp="$(date +%Y%m%d_%H%M%S)"

backup_file() {
  local path="$1"
  if [[ -e "$path" ]]; then
    cp -a -- "$path" "${path}.bak.${timestamp}"
    printf 'Backup: %s.bak.%s\n' "$path" "$timestamp"
  fi
}

remove_active_keys() {
  local path="$1"
  shift
  local key
  for key in "$@"; do
    sed -i -E "/^[[:space:]]*${key}([[:space:]]+|$)/d" "$path"
  done
}

for config in "$NX_SERVER_CONFIG" "$NX_NODE_CONFIG"; do
  if [[ ! -f "$config" ]]; then
    printf 'NoMachine configuration not found: %s\n' "$config" >&2
    exit 1
  fi
done

backup_file "$NX_SERVER_CONFIG"
backup_file "$NX_NODE_CONFIG"
backup_file "$TIME_UNIT"
backup_file "$GDM_TIME_DROPIN"
backup_file "$NX_DROPIN"
backup_file "$WAIT_HELPER"
backup_file "$GUARD_HELPER"
backup_file "$GUARD_UNIT"

sed -i '/^# BEGIN local physical-desktop-only policy$/,/^# END local physical-desktop-only policy$/d' \
  "$NX_SERVER_CONFIG"
remove_active_keys "$NX_SERVER_CONFIG" \
  ConnectPolicy AvailableSessionTypes VirtualDesktopAccess \
  EnableGuestCreateVirtualDesktop CreateDisplay StartNXDaemon \
  ClientConnectionMethods
cat >>"$NX_SERVER_CONFIG" <<'EOF'

# BEGIN local physical-desktop-only policy
ConnectPolicy autocreate=0,autoconnect=0,automigrate=0,desktop=0,dialog=0,xsessions=0,udp=1
AvailableSessionTypes physical-desktop
VirtualDesktopAccess none
EnableGuestCreateVirtualDesktop 0
CreateDisplay 0
StartNXDaemon Manual
ClientConnectionMethods NX
# END local physical-desktop-only policy
EOF

sed -i '/^# BEGIN local physical-desktop-only policy$/,/^# END local physical-desktop-only policy$/d' \
  "$NX_NODE_CONFIG"
remove_active_keys "$NX_NODE_CONFIG" \
  AvailableSessionTypes PhysicalDisplays ClientConnectionMethods
cat >>"$NX_NODE_CONFIG" <<'EOF'

# BEGIN local physical-desktop-only policy
AvailableSessionTypes physical-desktop
PhysicalDisplays :0-:9
ClientConnectionMethods NX
# END local physical-desktop-only policy
EOF

install -D -m 0755 "$PROJECT_ROOT/tools/systemd/nx-wait-physical-xorg.sh" "$WAIT_HELPER"
install -D -m 0755 "$PROJECT_ROOT/tools/systemd/nx-guard-physical-xorg.sh" "$GUARD_HELPER"
install -D -m 0644 "$PROJECT_ROOT/tools/systemd/nx-physical-xorg-guard.service" "$GUARD_UNIT"
install -d -m 0755 "$NX_DROPIN_DIR"
cat >"$NX_DROPIN" <<EOF
[Unit]
After=display-manager.service

[Service]
ExecStartPre=+$WAIT_HELPER 120
TimeoutStartSec=150s
KillMode=control-group
EOF

# Keep RTC synchronization available, but never put it in GDM's critical path.
if [[ -f "$TIME_UNIT" ]]; then
  sed -i \
    -e 's/^Description=.*/Description=Update RTC after network time becomes available/' \
    -e '/^[[:space:]]*Before=gdm\.service[[:space:]]*$/d' \
    "$TIME_UNIT"
fi
rm -f -- "$GDM_TIME_DROPIN"

systemctl daemon-reload
systemctl enable nx-physical-xorg-guard.service
install -m 0600 /dev/null "$RESTART_MARKER"

if ! "$WAIT_HELPER" 5; then
  printf 'Warning: no physical Xorg is ready yet; NoMachine will remain unavailable.\n' >&2
fi

printf '\nThe physical-Xorg-only NoMachine policy is installed.\n'
printf 'The runtime guard is enabled for the next boot but was not started now.\n'
printf 'After saving open applications, start it to remove the current virtual session:\n'
printf '  systemctl start nx-physical-xorg-guard.service\n'
