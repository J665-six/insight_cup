#!/usr/bin/env bash
set -Eeuo pipefail

if (( EUID != 0 )); then
  printf 'Run as root: sudo %s\n' "$0" >&2
  exit 2
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
timestamp="$(date +%Y%m%d_%H%M%S)"

backup_file() {
  local path="$1"
  if [[ -e "$path" ]]; then
    cp -a -- "$path" "${path}.bak.${timestamp}"
    printf 'Backup: %s.bak.%s\n' "$path" "$timestamp"
  fi
}

# Keep the NoMachine encoder and display server out of realtime/automatic mode.
"$PROJECT_ROOT/tools/systemd/configure-nomachine-safe.sh"
"$PROJECT_ROOT/tools/systemd/configure-nomachine-physical-only.sh"

XORG_CONFIG="${RK3588_XORG_CONFIG:-/etc/X11/xorg.conf.d/20-modesetting.conf}"
if [[ -f "$XORG_CONFIG" ]]; then
  backup_file "$XORG_CONFIG"
  sed -i \
    -e 's/^[[:space:]]*Option[[:space:]]*"AccelMethod"[[:space:]]*"glamor"/    Option      "AccelMethod"    "none"/' \
    -e '/^[[:space:]]*Option[[:space:]]*"DRI"[[:space:]]/d' \
    -e '/^[[:space:]]*Option[[:space:]]*"FlipFB"[[:space:]]/d' \
    -e '/^[[:space:]]*Option[[:space:]]*"NoEDID"[[:space:]]/d' \
    -e '/^[[:space:]]*Option[[:space:]]*"UseGammaLUT"[[:space:]]/d' \
    "$XORG_CONFIG"
  if ! grep -q '"ShadowFB"' "$XORG_CONFIG"; then
    sed -i '/^[[:space:]]*Option[[:space:]]*"AccelMethod"/a\    Option      "ShadowFB"       "true"' "$XORG_CONFIG"
  fi
else
  printf 'Xorg configuration not found: %s\n' "$XORG_CONFIG" >&2
fi

GDM_CONFIG="${RK3588_GDM_CONFIG:-/etc/gdm3/custom.conf}"
if [[ -f "$GDM_CONFIG" ]]; then
  backup_file "$GDM_CONFIG"
  if grep -qE '^[[:space:]]*#?[[:space:]]*WaylandEnable[[:space:]]*=' "$GDM_CONFIG"; then
    sed -i -E 's/^[[:space:]]*#?[[:space:]]*WaylandEnable[[:space:]]*=.*/WaylandEnable=false/' "$GDM_CONFIG"
  else
    printf '\n[daemon]\nWaylandEnable=false\n' >>"$GDM_CONFIG"
  fi
fi

# These vendor services fail forever on this image: the first-boot archive and
# bt_pcba_test are absent. They are unrelated to the functioning kernel drivers
# and only leave failed units and repeated boot-time errors behind.
for unit in rockchip.service rk-bt.service; do
  systemctl disable --now "$unit" >/dev/null 2>&1 || true
  systemctl reset-failed "$unit" >/dev/null 2>&1 || true
done

# fwupd ships an i2c_dev load request, but this vendor kernel has no module by
# that name. Hide only the invalid load request, preserving the original file.
MODULE_CONF="/usr/lib/modules-load.d/fwupd-i2c.conf"
if [[ -f "$MODULE_CONF" ]]; then
  mv -- "$MODULE_CONF" "${MODULE_CONF}.disabled.${timestamp}"
  printf 'Disabled unavailable module request: %s\n' "$MODULE_CONF"
fi
systemctl reset-failed systemd-modules-load.service >/dev/null 2>&1 || true

# Bound persistent journal growth; the previous journal had reached ~945 MiB.
JOURNAL_DROPIN=/etc/systemd/journald.conf.d/20-rk3588-limits.conf
install -d -m 0755 /etc/systemd/journald.conf.d
backup_file "$JOURNAL_DROPIN"
printf '%s\n' \
  '[Journal]' \
  'Storage=persistent' \
  'SystemMaxUse=200M' \
  'RuntimeMaxUse=50M' \
  'MaxRetentionSec=14day' \
  'Compress=yes' | tee "$JOURNAL_DROPIN" >/dev/null

# Update the RTC after ntpd has corrected it, without delaying GDM. NoMachine
# independently waits for a real Xorg socket before accepting connections.
TIME_UNIT=/etc/systemd/system/rk3588-time-sync.service
backup_file "$TIME_UNIT"
printf '%s\n' \
  '[Unit]' \
  'Description=Update RTC after network time becomes available' \
  'Wants=network-online.target' \
  'After=network-online.target ntp.service' \
  '' \
  '[Service]' \
  'Type=oneshot' \
  'ExecStart=/usr/sbin/ntp-wait -n 20 -s 3' \
  'ExecStartPost=/sbin/hwclock --systohc' \
  'TimeoutStartSec=70s' \
  'RemainAfterExit=yes' \
  '' \
  '[Install]' \
  'WantedBy=graphical.target' | tee "$TIME_UNIT" >/dev/null
rm -f -- /etc/systemd/system/gdm.service.d/10-rk3588-time-sync.conf

systemctl daemon-reload
systemctl enable ntp.service rk3588-time-sync.service >/dev/null
systemctl restart systemd-journald.service
journalctl --rotate >/dev/null 2>&1 || true
journalctl --vacuum-size=200M >/dev/null 2>&1 || true

printf '\nRK3588 desktop safety configuration applied.\n'
printf 'Xorg uses the software framebuffer; NTP no longer blocks GDM.\n'
printf 'Restart GDM to apply Xorg now (this disconnects graphical sessions):\n'
printf '  systemctl restart gdm.service\n'
