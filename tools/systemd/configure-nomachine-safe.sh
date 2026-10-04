#!/usr/bin/env bash
set -Eeuo pipefail

# NoMachine's physical-desktop defaults on this RK3588 image use realtime
# scheduling and an automatic encoder thread count. That combination can
# starve Xorg/GNOME when the recognition pipeline is also running.
if (( EUID != 0 )); then
    printf 'Run as root: sudo %s\n' "$0" >&2
    exit 2
fi

CONFIG="${NX_NODE_CONFIG:-/usr/NX/etc/node.cfg}"
if [[ ! -f "$CONFIG" ]]; then
    printf 'NoMachine node config not found: %s\n' "$CONFIG" >&2
    exit 1
fi

backup="${CONFIG}.bak.$(date +%Y%m%d_%H%M%S)"
cp -a -- "$CONFIG" "$backup"

# Remove active overrides while leaving the vendor comments intact.
sed -i \
    -e '/^[[:space:]]*DisplayServerThreads[[:space:]]/d' \
    -e '/^[[:space:]]*DisplayEncoderThreads[[:space:]]/d' \
    -e '/^[[:space:]]*DisplayServerPriority[[:space:]]/d' \
    -e '/^[[:space:]]*DisplayAgentPriority[[:space:]]/d' \
    -e '/^[[:space:]]*EnableEGLCapture[[:space:]]/d' \
    "$CONFIG"

cat >>"$CONFIG" <<'EOF'

# Local RK3588 stability limits for physical desktop sessions.
DisplayServerThreads 1
DisplayEncoderThreads 1
DisplayServerPriority normal
DisplayAgentPriority normal
EnableEGLCapture 0
EOF

printf 'Updated %s\nBackup: %s\n' "$CONFIG" "$backup"
printf 'Restart NoMachine to apply (this disconnects active sessions):\n'
printf '  systemctl restart nxserver.service\n'
