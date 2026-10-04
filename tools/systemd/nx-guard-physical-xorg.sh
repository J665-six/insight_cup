#!/usr/bin/env bash
set -Eeuo pipefail

if (( EUID != 0 )); then
  printf 'Run as root: sudo %s\n' "$0" >&2
  exit 2
fi

NX_SERVICE="${NX_SERVICE:-nxserver.service}"
NX_SERVER="${NX_SERVER:-/usr/NX/bin/nxserver}"
WAIT_HELPER="${NX_WAIT_HELPER:-/usr/local/sbin/nx-wait-physical-xorg}"
AUDIT_SECONDS="${NX_GUARD_AUDIT_SECONDS:-1}"
RESTART_MARKER="${NX_POLICY_RESTART_MARKER:-/run/nomachine-physical-policy-restart-required}"
XORG_PID_FILE="${NX_XORG_PID_FILE:-/run/nomachine-physical-xorg.pid}"
VIRTUAL_REMOVED=0

if [[ ! -x "$NX_SERVER" || ! -x "$WAIT_HELPER" ]]; then
  printf 'Required executable is missing: %s or %s\n' "$NX_SERVER" "$WAIT_HELPER" >&2
  exit 1
fi

close_connection_gate() {
  "$NX_SERVER" --stop nxd >/dev/null 2>&1 || true
}

nxd_is_listening() {
  local pid fd target inode exe

  while read -r pid; do
    [[ -n "$pid" ]] || continue
    exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
    [[ "$exe" == /usr/NX/bin/nxd ]] || continue
    for fd in "/proc/$pid/fd"/*; do
      target="$(readlink -- "$fd" 2>/dev/null || true)"
      [[ "$target" =~ ^socket:\[([0-9]+)\]$ ]] || continue
      inode="${BASH_REMATCH[1]}"
      if awk -v inode="$inode" '
        $4 == "0A" && $10 == inode { found = 1 }
        END { exit(found ? 0 : 1) }
      ' /proc/net/tcp /proc/net/tcp6; then
        return 0
      fi
    done
  done < <(pgrep -x nxd 2>/dev/null || true)

  return 1
}

trap close_connection_gate EXIT
trap 'exit 0' INT TERM HUP

# Never inherit an open listener from an unguarded or crashed predecessor.
close_connection_gate

list_virtual_sessions() {
  local cmdline pid exe arg session_id is_virtual
  local -a argv

  for cmdline in /proc/[0-9]*/cmdline; do
    pid="${cmdline#/proc/}"
    pid="${pid%/cmdline}"
    exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
    [[ "$exe" == /usr/NX/bin/nxserver.bin ]] || continue

    argv=()
    mapfile -d '' -t argv <"$cmdline" 2>/dev/null || continue
    session_id=''
    is_virtual=0
    for ((i = 0; i < ${#argv[@]}; i++)); do
      case "${argv[$i]}" in
        --virtualsession)
          is_virtual=1
          ;;
        --sessionid)
          if (( i + 1 < ${#argv[@]} )); then
            session_id="${argv[$((i + 1))]}"
          fi
          ;;
      esac
    done

    if (( is_virtual )); then
      printf '%s\t%s\n' "$pid" "$session_id"
    fi
  done
}

remove_virtual_sessions() {
  local pid session_id remaining=0

  VIRTUAL_REMOVED=0

  while IFS=$'\t' read -r pid session_id; do
    [[ -n "$pid" ]] || continue
    VIRTUAL_REMOVED=1
    printf 'Removing forbidden NoMachine virtual session (pid=%s, id=%s).\n' \
      "$pid" "${session_id:-unknown}"

    if [[ "$session_id" =~ ^[[:xdigit:]]{32}$ ]]; then
      "$NX_SERVER" --terminate "$session_id" || true
    else
      printf 'Refusing to signal pid %s without a valid NoMachine session id.\n' "$pid" >&2
      remaining=1
      continue
    fi

    for _ in {1..30}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$pid" 2>/dev/null; then
      if [[ "$session_id" =~ ^[[:xdigit:]]{32}$ ]]; then
        printf 'Forcibly removing virtual session id %s.\n' "$session_id" >&2
        "$NX_SERVER" --kill "$session_id" || true
        for _ in {1..20}; do
          kill -0 "$pid" 2>/dev/null || break
          sleep 0.1
        done
      fi
      if kill -0 "$pid" 2>/dev/null; then
        printf 'Virtual session pid %s did not stop cleanly.\n' "$pid" >&2
        remaining=1
      fi
    fi
  done < <(list_virtual_sessions)

  return "$remaining"
}

while true; do
  if NX_XORG_PID_FILE="$XORG_PID_FILE" "$WAIT_HELPER" 120 >/dev/null 2>&1; then
    xorg_pid="$(<"$XORG_PID_FILE")"
    state="$(systemctl is-active "$NX_SERVICE" 2>/dev/null || true)"
    if [[ "$state" == inactive || "$state" == failed ]]; then
      if systemctl is-enabled --quiet "$NX_SERVICE"; then
        printf 'Physical Xorg is available; starting %s.\n' "$NX_SERVICE"
        systemctl start "$NX_SERVICE" || true
      fi
    fi

    state="$(systemctl is-active "$NX_SERVICE" 2>/dev/null || true)"
    if [[ "$state" != active ]]; then
      sleep "$AUDIT_SECONDS"
      continue
    fi

    if [[ -e "$RESTART_MARKER" ]]; then
      printf 'Loading the staged physical-desktop-only NoMachine policy.\n'
      close_connection_gate
      remove_virtual_sessions || true
      if systemctl restart "$NX_SERVICE"; then
        rm -f -- "$RESTART_MARKER"
      fi
      sleep "$AUDIT_SECONDS"
      continue
    fi

    if ! remove_virtual_sessions; then
      printf 'Restarting %s to clear a stuck virtual session.\n' "$NX_SERVICE" >&2
      close_connection_gate
      systemctl restart "$NX_SERVICE" || true
      sleep "$AUDIT_SECONDS"
      continue
    fi
    if (( VIRTUAL_REMOVED )); then
      printf 'Restarting %s so it discovers only the physical display.\n' "$NX_SERVICE"
      close_connection_gate
      systemctl restart "$NX_SERVICE" || true
      sleep "$AUDIT_SECONDS"
      continue
    fi

    if ! NX_XORG_REQUIRED_PID="$xorg_pid" NX_XORG_CHECK_ONCE=1 \
      "$WAIT_HELPER" 1 >/dev/null 2>&1; then
      continue
    fi

    if ! nxd_is_listening; then
      printf 'Physical Xorg is verified; opening the NoMachine NX listener.\n'
      if ! "$NX_SERVER" --start nxd; then
        close_connection_gate
        systemctl stop "$NX_SERVICE" || true
        continue
      fi
    fi
    if ! NX_XORG_REQUIRED_PID="$xorg_pid" NX_XORG_CHECK_ONCE=1 \
      "$WAIT_HELPER" 1 >/dev/null 2>&1; then
      close_connection_gate
      systemctl stop "$NX_SERVICE" || true
      continue
    fi

    while kill -0 "$xorg_pid" 2>/dev/null; do
      sleep "$AUDIT_SECONDS" &
      timer_pid=$!
      tail --pid="$xorg_pid" -f -s 0.1 /dev/null >/dev/null 2>&1 &
      watcher_pid=$!
      wait -n "$timer_pid" "$watcher_pid" || true
      kill "$timer_pid" "$watcher_pid" 2>/dev/null || true
      wait "$timer_pid" "$watcher_pid" 2>/dev/null || true
      kill -0 "$xorg_pid" 2>/dev/null || break
      NX_XORG_REQUIRED_PID="$xorg_pid" NX_XORG_CHECK_ONCE=1 \
        "$WAIT_HELPER" 1 >/dev/null 2>&1 || break

      state="$(systemctl is-active "$NX_SERVICE" 2>/dev/null || true)"
      [[ "$state" == active ]] || break

      if ! remove_virtual_sessions; then
        break
      fi
      if (( VIRTUAL_REMOVED )); then
        break
      fi
      if ! nxd_is_listening; then
        "$NX_SERVER" --start nxd || break
        NX_XORG_REQUIRED_PID="$xorg_pid" NX_XORG_CHECK_ONCE=1 \
          "$WAIT_HELPER" 1 >/dev/null 2>&1 || break
      fi
    done

    printf 'Physical Xorg disappeared or the guard found a virtual session; closing NoMachine.\n'
    close_connection_gate
    remove_virtual_sessions || true
    systemctl stop "$NX_SERVICE" || true
  else
    close_connection_gate
    remove_virtual_sessions || true
    systemctl stop "$NX_SERVICE" || true
  fi
done
