#!/usr/bin/env python3
"""Keep the RK3588 board online through this computer's Ethernet port."""

from __future__ import annotations

import argparse
import fcntl
import ipaddress
import logging
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

import paramiko


LOG = logging.getLogger("rk3588-network-share")
logging.getLogger("paramiko").setLevel(logging.CRITICAL)
LOCK_PATH = "/tmp/rk3588-network-share.lock"


DEFAULTS = {
    "HOST_IFACE": "enp109s0",
    "HOST_SHARE_IP": "192.168.10.21",
    "BOARD_IP": "192.168.10.51",
    "BOARD_MAC": "c2:30:1b:ab:0d:d8",
    "BOARD_IFACE": "eth0",
    "BOARD_USER": "zzurm",
    "BOARD_PASSWORD": "",
    "BOARD_CONNECTION": "Wired connection 1",
    "RETRY_SECONDS": "30",
}


def load_config() -> dict[str, str]:
    values = dict(DEFAULTS)
    env_path = Path(__file__).with_name("rk3588-network-share.env")
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    for key in values:
        if key in os.environ:
            values[key] = os.environ[key]
    if not values["BOARD_PASSWORD"]:
        raise RuntimeError(f"missing BOARD_PASSWORD in {env_path}")
    return values


def command(*args: str, check: bool = True, timeout: int = 20) -> subprocess.CompletedProcess[str]:
    LOG.debug("run: %s", " ".join(shlex.quote(arg) for arg in args))
    result = subprocess.run(
        list(args),
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"{args[0]} failed ({result.returncode}): {detail}")
    return result


def ipv4_addresses(interface: str) -> set[str]:
    result = command("ip", "-4", "-o", "addr", "show", "dev", interface, check=False)
    addresses: set[str] = set()
    for line in result.stdout.splitlines():
        match = re.search(r"\binet\s+(\d+\.\d+\.\d+\.\d+/\d+)", line)
        if match:
            addresses.add(match.group(1))
    return addresses


def active_connection(interface: str) -> str:
    result = command("nmcli", "-g", "GENERAL.CONNECTION", "device", "show", interface, check=False)
    connection = result.stdout.strip()
    if connection and connection not in {"--", "(externally)"}:
        return connection
    return ""


def host_connection_needs_update(connection: str, share_ip: str) -> bool:
    result = command(
        "env",
        "LC_ALL=C",
        "nmcli",
        "-g",
        "connection.autoconnect,ipv4.method,ipv4.addresses,ipv4.gateway,ipv4.dns,"
        "ipv4.never-default,ipv6.method",
        "connection",
        "show",
        connection,
        check=False,
    )
    actual = result.stdout.splitlines()
    desired = [
        "yes",
        "shared",
        f"{share_ip}/24",
        "",
        "",
        "yes",
        "disabled",
    ]
    return result.returncode != 0 or actual != desired


def ensure_host_shared(config: dict[str, str]) -> str:
    interface = config["HOST_IFACE"]
    share_ip = config["HOST_SHARE_IP"]
    if not Path(f"/sys/class/net/{interface}").exists():
        raise RuntimeError(f"host interface {interface} is not present")

    connection = active_connection(interface)
    if not connection:
        connection = "RK3588 computer share"
        existing = command("nmcli", "-t", "-f", "NAME", "connection", "show", check=False).stdout
        if connection not in existing.splitlines():
            command(
                "nmcli",
                "connection",
                "add",
                "type",
                "ethernet",
                "ifname",
                interface,
                "con-name",
                connection,
            )

    if host_connection_needs_update(connection, share_ip):
        desired = [
            "connection.autoconnect",
            "yes",
            "ipv4.method",
            "shared",
            "ipv4.addresses",
            f"{share_ip}/24",
            "ipv4.gateway",
            "",
            "ipv4.dns",
            "",
            "ipv4.never-default",
            "yes",
            "ipv6.method",
            "disabled",
        ]
        command("nmcli", "connection", "modify", connection, *desired)
        command("nmcli", "device", "reapply", interface, check=False)

    if f"{share_ip}/24" not in ipv4_addresses(interface):
        command("nmcli", "connection", "up", connection, "ifname", interface, timeout=30)

    if f"{share_ip}/24" not in ipv4_addresses(interface):
        raise RuntimeError(f"shared address {share_ip}/24 is not active on {interface}")

    LOG.info("computer share ready: %s -> %s", interface, share_ip)
    return interface


def parse_neighbours(interface: str, wanted_mac: str) -> list[str]:
    result = command("ip", "neigh", "show", "dev", interface, check=False)
    wanted_mac = wanted_mac.lower()
    matches: list[str] = []
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 5 and fields[1] == "lladdr" and fields[2].lower() == wanted_mac:
            try:
                ipaddress.ip_address(fields[0])
                matches.append(fields[0])
            except ValueError:
                pass
    return matches


def dhcp_lease_addresses(wanted_mac: str) -> list[str]:
    lease_files = Path("/var/lib/NetworkManager").glob("dnsmasq-*.leases")
    wanted_mac = wanted_mac.lower()
    matches: list[str] = []
    for lease_file in lease_files:
        try:
            lines = lease_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) >= 3 and fields[1].lower() == wanted_mac:
                try:
                    ipaddress.ip_address(fields[2])
                    matches.append(fields[2])
                except ValueError:
                    pass
    return matches


def board_candidates(config: dict[str, str], host_interface: str) -> list[str]:
    candidates: list[str] = []

    def add(value: str) -> None:
        if value and value not in candidates:
            candidates.append(value)

    add(config["BOARD_IP"])
    for value in parse_neighbours(host_interface, config["BOARD_MAC"]):
        add(value)
    for value in dhcp_lease_addresses(config["BOARD_MAC"]):
        add(value)
    return candidates


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def connect_board(config: dict[str, str], host_interface: str) -> tuple[paramiko.SSHClient, str]:
    candidates = board_candidates(config, host_interface)
    LOG.debug("board candidates: %s", ", ".join(candidates) or "(none)")
    for candidate in candidates:
        if not port_open(candidate, 22, timeout=1.5):
            continue
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                candidate,
                port=22,
                username=config["BOARD_USER"],
                password=config["BOARD_PASSWORD"],
                timeout=5,
                banner_timeout=5,
                auth_timeout=5,
                look_for_keys=False,
                allow_agent=False,
            )
            LOG.info("connected to board SSH at %s", candidate)
            return client, candidate
        except (paramiko.SSHException, OSError) as error:
            LOG.warning("board SSH at %s is not ready: %s", candidate, error)
            client.close()
    raise RuntimeError("board SSH is not ready")


def sudo_script(client: paramiko.SSHClient, script: str, password: str) -> tuple[int, str, str]:
    remote_command = "sudo -S -p '' bash -lc " + shlex.quote(script)
    stdin, stdout, stderr = client.exec_command(remote_command, timeout=40)
    stdin.write(password + "\n")
    stdin.flush()
    status = stdout.channel.recv_exit_status()
    return status, stdout.read().decode(errors="replace"), stderr.read().decode(errors="replace")


def board_network_is_healthy(client: paramiko.SSHClient, config: dict[str, str]) -> bool:
    """Avoid touching a working board connection on every health-check cycle."""
    board_iface = shlex.quote(config["BOARD_IFACE"])
    host_ip = shlex.quote(config["HOST_SHARE_IP"])
    health_check = f"""
set -e
board_iface={board_iface}
host_ip={host_ip}
test -x /usr/local/sbin/rk3588-host-gateway
grep -Fq 'rk3588-host-gateway-v2' /usr/local/sbin/rk3588-host-gateway
systemctl is-enabled --quiet rk3588-host-gateway.timer
ip -4 -o addr show dev "$board_iface" scope global | awk '{{print $4}}' | grep -Fxq "{config["BOARD_IP"]}/24"
ip route get 1.1.1.1 | grep -Eq "via $host_ip dev $board_iface( |$)"
ping -c 1 -W 3 1.1.1.1 >/dev/null
getent hosts www.baidu.com >/dev/null
"""
    stdin, stdout, stderr = client.exec_command(health_check, timeout=15)
    status = stdout.channel.recv_exit_status()
    stdout.read()
    stderr.read()
    return status == 0


def configure_board(config: dict[str, str], client: paramiko.SSHClient) -> None:
    board_iface = shlex.quote(config["BOARD_IFACE"])
    board_ip = shlex.quote(config["BOARD_IP"])
    host_ip = shlex.quote(config["HOST_SHARE_IP"])
    configured_connection = shlex.quote(config["BOARD_CONNECTION"])
    gateway_script = f"""#!/bin/sh
set -u
export LC_ALL=C
# rk3588-host-gateway-v2: only reapply NetworkManager when the route drifts.
board_ip="{config["BOARD_IP"]}"
host_ip="{config["HOST_SHARE_IP"]}"

if systemctl list-unit-files dhcpcd.service >/dev/null 2>&1; then
    systemctl disable --now dhcpcd.service >/dev/null 2>&1 || true
fi

connection={configured_connection}
if ! nmcli -t -f NAME connection show | grep -Fxq "$connection"; then
    connection="$(nmcli -g GENERAL.CONNECTION device show {board_iface} 2>/dev/null || true)"
fi
if [ -z "$connection" ] || [ "$connection" = "--" ]; then
    connection="RK3588 board Ethernet"
    nmcli connection add type ethernet ifname {board_iface} con-name "$connection" >/dev/null
fi

keep_address="{config["BOARD_IP"]}/24"
needs_repair=0
if ! ip -4 -o addr show dev {board_iface} scope global | awk '{{print $4}}' | grep -Fxq "$keep_address"; then
    needs_repair=1
fi
if ! ip route get 1.1.1.1 2>/dev/null | grep -Eq "via {config["HOST_SHARE_IP"]} dev {config["BOARD_IFACE"]}( |$)"; then
    needs_repair=1
fi

if [ "$needs_repair" -eq 1 ]; then
    nmcli connection modify "$connection" \
        connection.autoconnect yes \
        ipv4.method manual \
        ipv4.addresses "{config["BOARD_IP"]}/24" \
        ipv4.gateway "{config["HOST_SHARE_IP"]}" \
        ipv4.dns "{config["HOST_SHARE_IP"]}" \
        ipv4.never-default no \
        ipv4.route-metric 50 \
        ipv6.method disabled
    if ip link show dev {board_iface} | grep -q "LOWER_UP"; then
        nmcli device reapply {board_iface} >/dev/null 2>&1 || nmcli connection up "$connection" ifname {board_iface} >/dev/null 2>&1 || true
    fi
fi

for address in $(ip -4 -o addr show dev {board_iface} scope global | awk -v keep="$keep_address" '$4 != keep {{ print $4 }}'); do
    ip addr del "$address" dev {board_iface} >/dev/null 2>&1 || true
done
ip route replace default via "{config["HOST_SHARE_IP"]}" dev {board_iface} metric 50 >/dev/null 2>&1 || true
resolvectl dns {board_iface} "{config["HOST_SHARE_IP"]}" >/dev/null 2>&1 || true
"""
    board_service = """[Unit]
Description=Repair RK3588 Ethernet gateway through the computer
After=NetworkManager.service
Wants=NetworkManager.service

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/rk3588-host-gateway
"""
    board_timer = """[Unit]
Description=Periodically repair RK3588 Ethernet gateway

[Timer]
OnBootSec=15s
OnUnitActiveSec=30s
Unit=rk3588-host-gateway.service

[Install]
WantedBy=timers.target
"""
    script = f"""
set -u
export LC_ALL=C
board_ip="{config["BOARD_IP"]}"
host_ip="{config["HOST_SHARE_IP"]}"

if systemctl list-unit-files dhcpcd.service >/dev/null 2>&1; then
    systemctl disable --now dhcpcd.service >/dev/null 2>&1 || true
fi

connection={configured_connection}
if ! nmcli -t -f NAME connection show | grep -Fxq "$connection"; then
    connection="$(nmcli -g GENERAL.CONNECTION device show {board_iface} 2>/dev/null || true)"
fi
if [ -z "$connection" ] || [ "$connection" = "--" ]; then
    connection="RK3588 board Ethernet"
    nmcli connection add type ethernet ifname {board_iface} con-name "$connection" >/dev/null
fi

keep_address="{config["BOARD_IP"]}/24"
needs_repair=0
if ! ip -4 -o addr show dev {board_iface} scope global | awk '{{print $4}}' | grep -Fxq "$keep_address"; then
    needs_repair=1
fi
if ! ip route get 1.1.1.1 2>/dev/null | grep -Eq "via {config["HOST_SHARE_IP"]} dev {config["BOARD_IFACE"]}( |$)"; then
    needs_repair=1
fi

if [ "$needs_repair" -eq 1 ]; then
    nmcli connection modify "$connection" \
        connection.autoconnect yes \
        ipv4.method manual \
        ipv4.addresses "$board_ip/24" \
        ipv4.gateway "$host_ip" \
        ipv4.dns "$host_ip" \
        ipv4.never-default no \
        ipv4.route-metric 50 \
        ipv6.method disabled
    if ip link show dev {board_iface} | grep -q "LOWER_UP"; then
        nmcli device reapply {board_iface} >/dev/null 2>&1 || nmcli connection up "$connection" ifname {board_iface} >/dev/null 2>&1 || true
    fi
fi

for address in $(ip -4 -o addr show dev {board_iface} scope global | awk -v keep="$keep_address" '$4 != keep {{ print $4 }}'); do
    ip addr del "$address" dev {board_iface} >/dev/null 2>&1 || true
done
ip route replace default via {host_ip} dev {board_iface} metric 50 >/dev/null 2>&1 || true
resolvectl dns {board_iface} {host_ip} >/dev/null 2>&1 || true

install -m 0755 /dev/stdin /usr/local/sbin/rk3588-host-gateway <<'RK3588_GATEWAY_SCRIPT'
{gateway_script}
RK3588_GATEWAY_SCRIPT
install -m 0644 /dev/stdin /etc/systemd/system/rk3588-host-gateway.service <<'RK3588_GATEWAY_SERVICE'
{board_service}
RK3588_GATEWAY_SERVICE
install -m 0644 /dev/stdin /etc/systemd/system/rk3588-host-gateway.timer <<'RK3588_GATEWAY_TIMER'
{board_timer}
RK3588_GATEWAY_TIMER
systemctl daemon-reload
systemctl enable --now rk3588-host-gateway.timer >/dev/null 2>&1 || true
"""
    status, stdout, stderr = sudo_script(client, script, config["BOARD_PASSWORD"])
    if status != 0:
        raise RuntimeError(f"board configuration failed: {(stderr or stdout).strip()}")
    LOG.info("board Ethernet configuration applied")


def verify_board(client: paramiko.SSHClient, config: dict[str, str]) -> None:
    verify = f"""
set -e
test "$(ip -4 -o addr show dev {shlex.quote(config["BOARD_IFACE"])} scope global | awk '{{print $4}}' | head -n1)" = "{config["BOARD_IP"]}/24"
test "$(ip route get 1.1.1.1 | awk '{{print $3; exit}}')" = "{config["HOST_SHARE_IP"]}"
ping -c 1 -W 3 1.1.1.1 >/dev/null
getent hosts www.baidu.com >/dev/null
"""
    status, stdout, stderr = sudo_script(client, verify, config["BOARD_PASSWORD"])
    if status != 0:
        raise RuntimeError(f"board verification failed: {(stderr or stdout).strip()}")
    LOG.info("board verified: gateway/DNS/internet are working")


def run_once(config: dict[str, str]) -> bool:
    host_interface = ensure_host_shared(config)
    try:
        client, board_address = connect_board(config, host_interface)
    except RuntimeError as error:
        LOG.warning("%s", error)
        return False

    try:
        if board_network_is_healthy(client, config):
            LOG.info("board network already healthy through %s", board_address)
            return True
        configure_board(config, client)
        verify_board(client, config)
        LOG.info("RK3588 network sharing is ready through %s", board_address)
        return True
    except (RuntimeError, paramiko.SSHException, OSError) as error:
        LOG.warning("board setup did not complete: %s", error)
        return False
    finally:
        client.close()


def run_daemon(config: dict[str, str]) -> int:
    retry_seconds = max(15, int(config["RETRY_SECONDS"]))
    LOG.info("RK3588 network sharing daemon started")
    while True:
        try:
            if run_once(config):
                time.sleep(60)
            else:
                time.sleep(retry_seconds)
        except Exception:
            LOG.exception("unexpected network-share error")
            time.sleep(retry_seconds)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="configure once and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = load_config()
    lock_file = open(LOCK_PATH, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        LOG.info("another network-share instance is already running")
        return 0

    if args.once:
        return 0 if run_once(config) else 1
    return run_daemon(config)


if __name__ == "__main__":
    sys.exit(main())
