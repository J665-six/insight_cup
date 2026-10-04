#!/usr/bin/env bash
set -Eeuo pipefail

staged=0
if [[ "${1:-}" == --staged ]]; then
  staged=1
elif (( $# > 0 )); then
  printf 'Usage: %s [--staged]\n' "$0" >&2
  exit 2
fi

# Return success when NoMachine is not installed. This keeps the guard useful
# on headless machines and in CI while failing closed on the affected image.
CONFIG="${NX_NODE_CONFIG:-/usr/NX/etc/node.cfg}"
SERVER_CONFIG="${NX_SERVER_CONFIG:-/usr/NX/etc/server.cfg}"
if [[ ! -e "$CONFIG" ]]; then
  exit 0
fi
if [[ ! -r "$CONFIG" ]]; then
  printf 'Cannot read NoMachine configuration: %s\n' "$CONFIG" >&2
  exit 1
fi

declare -A expected=(
  [DisplayServerThreads]=1
  [DisplayEncoderThreads]=1
  [DisplayServerPriority]=normal
  [DisplayAgentPriority]=normal
  [EnableEGLCapture]=0
)
failed=0
config_value() {
  local path="$1" key="$2"
  awk -v key="$key" '$1 == key { value = $2 } END { print value }' "$path"
}

for key in "${!expected[@]}"; do
  value="$(config_value "$CONFIG" "$key")"
  if [[ "$value" != "${expected[$key]}" ]]; then
    printf 'Unsafe NoMachine setting: %s=%s (expected %s)\n' \
      "$key" "${value:-<unset>}" "${expected[$key]}" >&2
    failed=1
  fi
done

headless_nomachine_ok() {
  local failed=0 connect_policy
  local session_types='unix-remote,unix-console,unix-default,unix-application,physical-desktop,shadow,unix-xsession-default,unix-gnome,unix-xdm'
  local expected_policy='autocreate=1,autoconnect=1,automigrate=0,desktop=1,dialog=0,xsessions=1,udp=1'

  [[ -r "$SERVER_CONFIG" ]] || return 1

  [[ "$(config_value "$CONFIG" AvailableSessionTypes)" == "$session_types" ]] || failed=1
  [[ "$(config_value "$CONFIG" ClientConnectionMethods)" == NX ]] || failed=1
  [[ "$(config_value "$SERVER_CONFIG" AvailableSessionTypes)" == "$session_types" ]] || failed=1
  [[ "$(config_value "$SERVER_CONFIG" VirtualDesktopAccess)" == administrator,trusted,owner,system ]] || failed=1
  [[ "$(config_value "$SERVER_CONFIG" EnableGuestCreateVirtualDesktop)" == 0 ]] || failed=1
  [[ "$(config_value "$SERVER_CONFIG" CreateDisplay)" == 1 ]] || failed=1
  [[ "$(config_value "$SERVER_CONFIG" StartNXDaemon)" == Automatic ]] || failed=1
  [[ "$(config_value "$SERVER_CONFIG" ClientConnectionMethods)" == NX ]] || failed=1

  connect_policy="$(config_value "$SERVER_CONFIG" ConnectPolicy)"
  [[ "$connect_policy" == "$expected_policy" ]] || failed=1

  if (( ! staged )); then
    if systemctl is-active --quiet gdm3.service 2>/dev/null ||
       pgrep -f '^/usr/lib/xorg/Xorg .*[[:space:]]vt[0-9]+' >/dev/null; then
      failed=1
    fi
    if ! systemctl is-active --quiet nxserver.service 2>/dev/null; then
      failed=1
    fi
    if ! pgrep -f '^/usr/NX/bin/nxserver\.bin --virtualsession( |$)' >/dev/null; then
      failed=1
    fi
    if ! ss -ltn 2>/dev/null | awk '$4 ~ /(^|:)(4000)$/ { found = 1 } END { exit(found ? 0 : 1) }'; then
      failed=1
    fi
  fi

  return "$failed"
}

# The RK3588 can be safe in two different ways:
# 1. the original physical-desktop-only policy below, or
# 2. a headless NoMachine nxvfb desktop that never touches Rockchip DRM/Xorg.
if (( ! failed )) && headless_nomachine_ok; then
  exit 0
fi

declare -A physical_only_node=(
  [AvailableSessionTypes]=physical-desktop
  [PhysicalDisplays]=:0-:9
)
for key in "${!physical_only_node[@]}"; do
  value="$(config_value "$CONFIG" "$key")"
  if [[ "$value" != "${physical_only_node[$key]}" ]]; then
    printf 'Unsafe NoMachine setting: %s=%s (expected %s)\n' \
      "$key" "${value:-<unset>}" "${physical_only_node[$key]}" >&2
    failed=1
  fi
done

if [[ ! -r "$SERVER_CONFIG" ]]; then
  printf 'Cannot read NoMachine configuration: %s\n' "$SERVER_CONFIG" >&2
  failed=1
else
  declare -A physical_only_server=(
    [AvailableSessionTypes]=physical-desktop
    [VirtualDesktopAccess]=none
    [EnableGuestCreateVirtualDesktop]=0
    [CreateDisplay]=0
    [StartNXDaemon]=Manual
    [ClientConnectionMethods]=NX
  )
  for key in "${!physical_only_server[@]}"; do
    value="$(config_value "$SERVER_CONFIG" "$key")"
    if [[ "$value" != "${physical_only_server[$key]}" ]]; then
      printf 'Unsafe NoMachine setting: %s=%s (expected %s)\n' \
        "$key" "${value:-<unset>}" "${physical_only_server[$key]}" >&2
      failed=1
    fi
  done

  connect_policy="$(config_value "$SERVER_CONFIG" ConnectPolicy)"
  expected_policy='autocreate=0,autoconnect=0,automigrate=0,desktop=0,dialog=0,xsessions=0,udp=1'
  if [[ "$connect_policy" != "$expected_policy" ]]; then
    printf 'Unsafe NoMachine setting: ConnectPolicy=%s (expected %s)\n' \
      "${connect_policy:-<unset>}" "$expected_policy" >&2
    failed=1
  fi
fi

node_connection_methods="$(config_value "$CONFIG" ClientConnectionMethods)"
if [[ "$node_connection_methods" != NX ]]; then
  printf 'Unsafe NoMachine setting: ClientConnectionMethods=%s in node config (expected NX)\n' \
    "${node_connection_methods:-<unset>}" >&2
  failed=1
fi

WAIT_HELPER="${NX_WAIT_HELPER:-/usr/local/sbin/nx-wait-physical-xorg}"
GUARD_HELPER="${NX_GUARD_HELPER:-/usr/local/sbin/nx-guard-physical-xorg}"
NX_DROPIN="${NX_DROPIN:-/etc/systemd/system/nxserver.service.d/10-physical-xorg.conf}"
GUARD_UNIT="${NX_GUARD_UNIT:-/etc/systemd/system/nx-physical-xorg-guard.service}"

for executable in "$WAIT_HELPER" "$GUARD_HELPER"; do
  if [[ ! -x "$executable" ]]; then
    printf 'Missing executable physical-desktop guard: %s\n' "$executable" >&2
    failed=1
  fi
done

if [[ ! -r "$NX_DROPIN" ]] || ! grep -Fq "ExecStartPre=+$WAIT_HELPER 120" "$NX_DROPIN"; then
  printf 'Missing NoMachine physical-Xorg startup gate: %s\n' "$NX_DROPIN" >&2
  failed=1
fi
if [[ ! -r "$GUARD_UNIT" ]] || ! grep -Fq "ExecStart=$GUARD_HELPER" "$GUARD_UNIT"; then
  printf 'Missing NoMachine physical-Xorg runtime guard: %s\n' "$GUARD_UNIT" >&2
  failed=1
fi
if ! systemctl is-enabled --quiet nx-physical-xorg-guard.service 2>/dev/null; then
  printf 'NoMachine physical-Xorg runtime guard is not enabled.\n' >&2
  failed=1
fi

if (( ! staged )); then
  if ! systemctl is-active --quiet nx-physical-xorg-guard.service 2>/dev/null; then
    printf 'NoMachine physical-Xorg runtime guard is not active.\n' >&2
    failed=1
  fi
  if pgrep -f '^/usr/NX/bin/nxserver\.bin --virtualsession( |$)' >/dev/null; then
    printf 'A forbidden NoMachine virtual session is still running.\n' >&2
    failed=1
  fi
fi

if (( failed )); then
  printf '\nRun the following as root, then restart NoMachine:\n' >&2
  printf '  sudo %q\n' "$(dirname -- "$0")/configure-nomachine-safe.sh" >&2
  printf '  sudo %q\n' "$(dirname -- "$0")/configure-nomachine-physical-only.sh" >&2
  printf '  sudo systemctl start nx-physical-xorg-guard.service\n' >&2
  exit 1
fi
