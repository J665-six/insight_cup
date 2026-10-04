#!/usr/bin/env bash
set -Eeuo pipefail

timeout_seconds="${1:-120}"
if [[ ! "$timeout_seconds" =~ ^[0-9]+$ ]] || (( timeout_seconds < 1 || timeout_seconds > 600 )); then
  printf 'Usage: %s [timeout-seconds: 1..600]\n' "$0" >&2
  exit 2
fi

stable_seconds="${NX_XORG_STABLE_SECONDS:-2}"
if [[ ! "$stable_seconds" =~ ^[0-9]+$ ]] || (( stable_seconds > 30 )); then
  printf 'NX_XORG_STABLE_SECONDS must be an integer from 0 to 30.\n' >&2
  exit 2
fi

pid_file="${NX_XORG_PID_FILE:-}"
required_pid="${NX_XORG_REQUIRED_PID:-}"
if [[ -n "$required_pid" && ! "$required_pid" =~ ^[0-9]+$ ]]; then
  printf 'NX_XORG_REQUIRED_PID must be a numeric process id.\n' >&2
  exit 2
fi

physical_xorg_pid() {
  local pid fd target inode exe cmdline uid

  while read -r pid; do
    [[ -n "$pid" && -d "/proc/$pid/fd" ]] || continue
    [[ -z "$required_pid" || "$pid" == "$required_pid" ]] || continue
    exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
    uid="$(awk '/^Uid:/ { print $3; exit }' "/proc/$pid/status" 2>/dev/null || true)"
    cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
    [[ "$exe" == /usr/lib/xorg/Xorg && "$uid" == 0 ]] || continue
    [[ "$cmdline" =~ (^|[[:space:]])vt[0-9]+([[:space:]]|$) ]] || continue

    for fd in "/proc/$pid/fd"/*; do
      target="$(readlink -- "$fd" 2>/dev/null || true)"
      if [[ "$target" =~ ^socket:\[([0-9]+)\]$ ]]; then
        inode="${BASH_REMATCH[1]}"
        if awk -v inode="$inode" '
          $6 == "01" && $7 == inode && $8 ~ "^/tmp/\\.X11-unix/X[0-9]$" {
            found = 1
          }
          END { exit(found ? 0 : 1) }
        ' /proc/net/unix; then
          printf '%s\n' "$pid"
          return 0
        fi
      fi
    done
  done < <(pgrep -x Xorg 2>/dev/null || true)

  return 1
}

if [[ "${NX_XORG_CHECK_ONCE:-0}" == 1 ]]; then
  pid="$(physical_xorg_pid || true)"
  if [[ -n "$pid" ]]; then
    printf 'Physical Xorg is ready (pid=%s).\n' "$pid"
    exit 0
  fi
  exit 1
fi

deadline=$((SECONDS + timeout_seconds))
candidate_pid=''
stable_since=0
while (( SECONDS < deadline )); do
  pid="$(physical_xorg_pid || true)"
  if [[ -n "$pid" ]]; then
    if [[ "$pid" != "$candidate_pid" ]]; then
      candidate_pid="$pid"
      stable_since=$SECONDS
    fi
    if (( SECONDS - stable_since >= stable_seconds )); then
      if [[ -n "$pid_file" ]]; then
        printf '%s\n' "$pid" >"$pid_file"
      fi
      printf 'Physical Xorg is ready (pid=%s, stable=%ss).\n' "$pid" "$stable_seconds"
      exit 0
    fi
  else
    candidate_pid=''
    stable_since=0
  fi

  sleep 0.25
done

printf 'Timed out after %ss waiting for a physical Xorg socket.\n' "$timeout_seconds" >&2
exit 1
