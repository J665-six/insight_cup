#!/usr/bin/env python3
"""Run repeatable UI/no-UI offline benchmarks on the RK3588 target."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = PROJECT_ROOT.parent
NPU_POWER_ROOT = Path("/sys/devices/platform/fdab0000.npu/power")
NPU_DEVFREQ_ROOT = Path("/sys/class/devfreq/fdab0000.npu")
NPU_DEBUG_LOAD = Path("/sys/kernel/debug/rknpu/load")
CLK_TCK = os.sysconf("SC_CLK_TCK")
PAGE_KIB = os.sysconf("SC_PAGE_SIZE") / 1024.0
LOGICAL_CPUS = os.cpu_count() or 1


@dataclass(frozen=True)
class VideoSpec:
    key: str
    filename: str
    expected_frames: int
    expected_fps: float
    expected_sha256: str


VIDEOS = {
    "all": VideoSpec(
        "all",
        "all_d455_color_20260917_230330_750162.mp4",
        885,
        30.0,
        "7cf42a4181a7c9037e216fb5917e95860171cf6005fdeaf37b75739c7642349e",
    ),
    "f1_f4": VideoSpec(
        "f1_f4",
        "f1_f4_d455_color_20260918_093215_640581.mp4",
        1197,
        30.0,
        "b995725ed87deb5e3bb46da92a89304df0863723b56ab282696d4105e775a524",
    ),
    "k4_k5_k10": VideoSpec(
        "k4_k5_k10",
        "k4_k5_k10_d455_color_20260918_081653_413685.mp4",
        1373,
        30.0,
        "2fc845b5a01a2aac0c185fd610855fc9a23d0ca9386ecbf6f796a7d44bd907ee",
    ),
}

# Alternating order limits systematic temperature and desktop-load bias.
DEFAULT_RUN_ORDER = (
    ("all", False),
    ("all", True),
    ("f1_f4", True),
    ("f1_f4", False),
    ("k4_k5_k10", False),
    ("k4_k5_k10", True),
)

RESOURCE_FIELDS = (
    "t_s",
    "phase",
    "system_cpu_pct",
    "app_cpu_core_pct",
    "app_cpu_machine_pct",
    "app_cpu_core_seconds_delta",
    "app_cpu_window_start_s",
    "app_cpu_window_end_s",
    "browser_cpu_core_pct",
    "browser_cpu_machine_pct",
    "browser_cpu_core_seconds_delta",
    "browser_cpu_window_start_s",
    "browser_cpu_window_end_s",
    "app_rss_kib",
    "browser_rss_kib",
    "app_threads",
    "browser_threads",
    "npu_debug_available",
    "npu_core0_pct",
    "npu_core1_pct",
    "npu_core2_pct",
    "npu_debug_mean_pct",
    "npu_power_active_pct",
    "npu_runtime_status",
    "npu_freq_hz",
    "npu_devfreq_load_invalid_pct",
    "soc_temp_c",
    "npu_temp_c",
)


def parse_args() -> argparse.Namespace:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(
        description="Benchmark three offline videos with and without a visible UI"
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/benchmarks" / f"offline_ui_{timestamp}",
    )
    parser.add_argument(
        "--binary", type=Path, default=PROJECT_ROOT / "build/insight-cup"
    )
    parser.add_argument("--sample-interval", type=float, default=0.2)
    parser.add_argument("--baseline-seconds", type=float, default=3.0)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional smoke-test frame limit; omit for the formal full-video run",
    )
    parser.add_argument("--port", type=int, default=17860)
    parser.add_argument(
        "--only",
        action="append",
        choices=[
            f"{video}-{mode}"
            for video in VIDEOS
            for mode in ("no-ui", "ui")
        ],
        help="Run only selected case(s); may be repeated",
    )
    parser.add_argument(
        "--skip-build", action="store_true", help="Do not run the one-time CMake build"
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Skip the known-input SHA-256 verification",
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def read_int(path: Path) -> int | None:
    value = read_text(path)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


@dataclass(frozen=True)
class ProcessStat:
    pid: int
    pgrp: int
    state: str
    ticks: int
    start_ticks: int
    rss_kib: float
    threads: int


def stat_fields(path: Path) -> list[str] | None:
    try:
        value = path.read_text(encoding="ascii")
    except (OSError, UnicodeError):
        return None
    end = value.rfind(")")
    return value[end + 2 :].split() if end >= 0 else None


def process_stat(pid: int) -> ProcessStat | None:
    fields = stat_fields(Path(f"/proc/{pid}/stat"))
    if fields is None:
        return None
    try:
        return ProcessStat(
            pid=pid,
            pgrp=int(fields[2]),
            state=fields[0],
            ticks=int(fields[11]) + int(fields[12]),
            start_ticks=int(fields[19]),
            rss_kib=int(fields[21]) * PAGE_KIB,
            threads=int(fields[17]),
        )
    except (IndexError, ValueError):
        return None


def process_group_stats(pgrp: int | None) -> list[ProcessStat]:
    if pgrp is None:
        return []
    output: list[ProcessStat] = []
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return output
    for entry in entries:
        if not entry.name.isdigit():
            continue
        stat = process_stat(int(entry.name))
        if stat is not None and stat.pgrp == pgrp and stat.state != "Z":
            output.append(stat)
    return output


def process_snapshot() -> list[ProcessStat]:
    output: list[ProcessStat] = []
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return output
    for entry in entries:
        if not entry.name.isdigit():
            continue
        stat = process_stat(int(entry.name))
        if stat is not None and stat.state != "Z":
            output.append(stat)
    return output


def system_ticks() -> tuple[int, int] | None:
    try:
        fields = Path("/proc/stat").read_text(encoding="ascii").splitlines()[0].split()
        values = [int(value) for value in fields[1:]]
    except (OSError, UnicodeError, IndexError, ValueError):
        return None
    total = sum(values)
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return total, idle


class ResourceSampler:
    def __init__(self, started: float) -> None:
        self.started = started
        self.previous_system = system_ticks()
        self.previous_processes: dict[str, dict[tuple[int, int], int]] = {}
        self.previous_process_wall: dict[str, float] = {}
        self.previous_npu_active = read_int(NPU_POWER_ROOT / "runtime_active_time")
        self.previous_npu_suspended = read_int(
            NPU_POWER_ROOT / "runtime_suspended_time"
        )

    def _process_values(
        self,
        name: str,
        pgrp: int | None,
        snapshot: Sequence[ProcessStat],
        sampled_at: float,
    ) -> tuple[
        float | None,
        float,
        int,
        float | None,
        float | None,
        float | None,
    ]:
        stats = [item for item in snapshot if item.pgrp == pgrp] if pgrp else []
        current = {(item.pid, item.start_ticks): item.ticks for item in stats}
        previous = self.previous_processes.get(name, {})
        previous_wall = self.previous_process_wall.get(name)
        elapsed = sampled_at - previous_wall if previous_wall is not None else 0.0
        delta_ticks = sum(
            max(0, ticks - previous[key])
            for key, ticks in current.items()
            if key in previous
        )
        self.previous_processes[name] = current
        self.previous_process_wall[name] = sampled_at
        valid_delta = elapsed > 0 and bool(previous) and bool(current)
        core_seconds_delta = delta_ticks / CLK_TCK if valid_delta else None
        cpu = (
            core_seconds_delta / elapsed * 100.0
            if core_seconds_delta is not None
            else None
        )
        return (
            cpu,
            sum(item.rss_kib for item in stats),
            sum(item.threads for item in stats),
            core_seconds_delta,
            previous_wall - self.started if valid_delta else None,
            sampled_at - self.started if valid_delta else None,
        )

    @staticmethod
    def _debug_npu_load() -> tuple[bool, list[float | None]]:
        value = read_text(NPU_DEBUG_LOAD)
        if value is None:
            return False, [None, None, None]
        cores: list[float | None] = [None, None, None]
        for core, load in re.findall(r"Core\s*([0-2])\s*:\s*([0-9.]+)%", value):
            cores[int(core)] = float(load)
        if all(item is None for item in cores):
            numbers = re.findall(r"(?:^|\s)([0-9.]+)%", value)
            for index, load in enumerate(numbers[:3]):
                cores[index] = float(load)
        return True, cores

    def sample(
        self, phase: str, app_pgrp: int | None, browser_pgrp: int | None
    ) -> dict[str, Any]:
        now = time.monotonic()
        current_system = system_ticks()
        snapshot = process_snapshot()
        processes_sampled_at = time.monotonic()
        system_cpu: float | None = None
        if current_system is not None and self.previous_system is not None:
            delta_total = current_system[0] - self.previous_system[0]
            delta_idle = current_system[1] - self.previous_system[1]
            if delta_total > 0:
                system_cpu = (delta_total - delta_idle) / delta_total * 100.0
        self.previous_system = current_system

        (
            app_cpu,
            app_rss,
            app_threads,
            app_core_seconds,
            app_window_start,
            app_window_end,
        ) = self._process_values(
            "app", app_pgrp, snapshot, processes_sampled_at
        )
        (
            browser_cpu,
            browser_rss,
            browser_threads,
            browser_core_seconds,
            browser_window_start,
            browser_window_end,
        ) = self._process_values(
            "browser", browser_pgrp, snapshot, processes_sampled_at
        )

        active = read_int(NPU_POWER_ROOT / "runtime_active_time")
        suspended = read_int(NPU_POWER_ROOT / "runtime_suspended_time")
        npu_power_active: float | None = None
        if (
            active is not None
            and suspended is not None
            and self.previous_npu_active is not None
            and self.previous_npu_suspended is not None
        ):
            delta_active = max(0, active - self.previous_npu_active)
            delta_suspended = max(0, suspended - self.previous_npu_suspended)
            if delta_active + delta_suspended > 0:
                npu_power_active = (
                    delta_active / (delta_active + delta_suspended) * 100.0
                )
        self.previous_npu_active = active
        self.previous_npu_suspended = suspended

        debug_available, npu_cores = self._debug_npu_load()
        available_cores = [item for item in npu_cores if item is not None]
        devfreq_value = read_text(NPU_DEVFREQ_ROOT / "load") or ""
        devfreq_match = re.match(r"([0-9.]+)@", devfreq_value)

        return {
            "t_s": round(now - self.started, 6),
            "phase": phase,
            "system_cpu_pct": system_cpu,
            "app_cpu_core_pct": app_cpu,
            "app_cpu_machine_pct": (
                app_cpu / LOGICAL_CPUS if app_cpu is not None else None
            ),
            "app_cpu_core_seconds_delta": app_core_seconds,
            "app_cpu_window_start_s": app_window_start,
            "app_cpu_window_end_s": app_window_end,
            "browser_cpu_core_pct": browser_cpu,
            "browser_cpu_machine_pct": (
                browser_cpu / LOGICAL_CPUS if browser_cpu is not None else None
            ),
            "browser_cpu_core_seconds_delta": browser_core_seconds,
            "browser_cpu_window_start_s": browser_window_start,
            "browser_cpu_window_end_s": browser_window_end,
            "app_rss_kib": app_rss,
            "browser_rss_kib": browser_rss,
            "app_threads": app_threads,
            "browser_threads": browser_threads,
            "npu_debug_available": int(debug_available),
            "npu_core0_pct": npu_cores[0],
            "npu_core1_pct": npu_cores[1],
            "npu_core2_pct": npu_cores[2],
            "npu_debug_mean_pct": (
                sum(available_cores) / len(available_cores)
                if available_cores
                else None
            ),
            "npu_power_active_pct": npu_power_active,
            "npu_runtime_status": read_text(NPU_POWER_ROOT / "runtime_status"),
            "npu_freq_hz": read_int(NPU_DEVFREQ_ROOT / "cur_freq"),
            # This RK3588 driver reports 100% even while runtime-suspended.
            # Retain the raw value for diagnosis but never use it as utilization.
            "npu_devfreq_load_invalid_pct": (
                float(devfreq_match.group(1)) if devfreq_match else None
            ),
            "soc_temp_c": temperature_c(0),
            "npu_temp_c": temperature_c(6),
        }


def temperature_c(zone: int) -> float | None:
    value = read_int(Path(f"/sys/class/thermal/thermal_zone{zone}/temp"))
    return value / 1000.0 if value is not None else None


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return value


class AppLogReader(threading.Thread):
    def __init__(self, stream: Any, destination: Path, started: float) -> None:
        super().__init__(daemon=True)
        self.stream = stream
        self.destination = destination
        self.started = started
        self.first_frame_t_s: float | None = None
        self.last_frame_t_s: float | None = None
        self.end_t_s: float | None = None
        self.frame_records = 0
        self.ui_state_request_seen = threading.Event()
        self.ui_stream_request_seen = threading.Event()

    def run(self) -> None:
        with self.destination.open("w", encoding="utf-8") as output:
            for line in iter(self.stream.readline, ""):
                output.write(line)
                output.flush()
                if self.first_frame_t_s is None and " | DEBUG   | FRAME " in line:
                    self.first_frame_t_s = time.monotonic() - self.started
                if " | DEBUG   | FRAME " in line:
                    self.frame_records += 1
                    self.last_frame_t_s = time.monotonic() - self.started
                if " | INFO    | END " in line:
                    self.end_t_s = time.monotonic() - self.started
                if "HTTP GET /api/state 200" in line:
                    self.ui_state_request_seen.set()
                if "HTTP GET /api/stream.mjpg 200" in line:
                    self.ui_stream_request_seen.set()
        self.stream.close()


def health_status(port: int) -> str | None:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/healthz", timeout=0.1
        ) as response:
            return str(json.loads(response.read()).get("status"))
    except Exception:
        return None


def post_control(port: int, action: str) -> bool:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/control",
        data=json.dumps({"action": action}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=0.5) as response:
            payload = json.loads(response.read())
            return response.status == 200 and payload.get("accepted") is True
    except Exception:
        return False


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def terminate_group(pgrp: int | None, grace_seconds: float = 4.0) -> None:
    if pgrp is None or pgrp <= 1:
        return
    try:
        os.killpg(pgrp, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not process_group_stats(pgrp):
            return
        time.sleep(0.1)
    try:
        os.killpg(pgrp, signal.SIGKILL)
    except ProcessLookupError:
        pass


def browser_command(profile: Path, port: int) -> list[str]:
    chromium = shutil.which("chromium") or shutil.which("chromium-browser")
    if chromium is None:
        raise RuntimeError("Visible UI benchmark requires Chromium")
    return [
        chromium,
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "--password-store=basic",
        "--disable-sync",
        "--disable-extensions",
        "--disable-default-apps",
        "--disable-background-mode",
        "--disable-component-update",
        "--window-size=1280,900",
        f"--app=http://127.0.0.1:{port}/",
    ]


def application_command(
    binary: Path,
    video: Path,
    output_root: Path,
    session_id: str,
    ui: bool,
    port: int,
    max_frames: int | None,
) -> list[str]:
    command = [
        str(binary),
        "--provider",
        "rknn",
        "--source",
        str(video),
        "--vid-stride",
        "1",
        "--stage2-workers",
        "1",
        "--shadow-log",
        "--verbose",
        "--frame-log-every",
        "1",
        "--no-save-video",
        "--no-save-crops",
        "--output-root",
        str(output_root),
        "--session-id",
        session_id,
    ]
    if max_frames is not None:
        command += ["--max-frames", str(max_frames)]
    if ui:
        command += [
            "--ui",
            "--no-open-browser",
            "--exit-on-complete",
            "--start-paused",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
    else:
        command.append("--no-ui")
    return command


def benchmark_environment(performance_output: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OMP_WAIT_POLICY": "PASSIVE",
            "KMP_BLOCKTIME": "0",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OPENCV_FOR_THREADS_NUM": "1",
            "INSIGHT_CUP_OPENCV_THREADS": "1",
            "INSIGHT_CUP_RKNN_PERF_OUTPUT": str(performance_output),
        }
    )
    return environment


def write_resource_row(writer: csv.DictWriter, row: dict[str, Any]) -> None:
    writer.writerow({field: csv_value(row.get(field)) for field in RESOURCE_FIELDS})


def run_case(
    spec: VideoSpec,
    ui: bool,
    args: argparse.Namespace,
    result_root: Path,
) -> dict[str, Any]:
    mode = "ui" if ui else "no-ui"
    run_id = f"{spec.key}-{mode}"
    run_root = result_root / "runs" / run_id
    run_root.mkdir(parents=True)
    sessions_root = result_root / "sessions"
    sessions_root.mkdir(parents=True, exist_ok=True)
    video = args.input_root.resolve() / spec.filename
    expected_frames = (
        min(spec.expected_frames, args.max_frames)
        if args.max_frames is not None
        else spec.expected_frames
    )
    case_port = args.port + list(VIDEOS).index(spec.key)
    command = application_command(
        args.binary.resolve(),
        video,
        sessions_root,
        run_id,
        ui,
        case_port,
        args.max_frames,
    )
    started_wall = datetime.now().astimezone().isoformat(timespec="milliseconds")
    started = time.monotonic()
    app: subprocess.Popen[str] | None = None
    browser: subprocess.Popen[str] | None = None
    browser_pgrp: int | None = None
    browser_started_t_s: float | None = None
    ui_paused_t_s: float | None = None
    ui_resumed_t_s: float | None = None
    browser_log_stream: Any = None
    profile = run_root / "chromium-profile"
    profile.mkdir()
    log_reader: AppLogReader | None = None
    timed_out = False
    app_returncode: int | None = None
    browser_returncode_at_app_exit: int | None = None
    browser_exited_early = False
    case_error: Exception | None = None
    sampler = ResourceSampler(started)
    resources_path = run_root / "resources.csv"
    meta_path = run_root / "run_meta.json"

    atomic_json(
        meta_path,
        {
            "schema": "insight_cup_offline_benchmark_run.v1",
            "run_id": run_id,
            "status": "running",
            "video": str(video),
            "ui": ui,
            "visible_browser": ui,
            "command": command,
            "started_at": started_wall,
        },
    )

    try:
        with resources_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESOURCE_FIELDS)
            writer.writeheader()
            next_sample = time.monotonic()
            baseline_deadline = started + args.baseline_seconds
            while time.monotonic() < baseline_deadline:
                now = time.monotonic()
                if now >= next_sample:
                    write_resource_row(writer, sampler.sample("baseline", None, None))
                    stream.flush()
                    next_sample += args.sample_interval
                time.sleep(min(0.02, max(0.0, next_sample - time.monotonic())))

            if ui and not port_is_free(case_port):
                raise RuntimeError(f"UI port {case_port} is already in use")
            app = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=benchmark_environment(run_root / "rknn_perf.jsonl"),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            assert app.stdout is not None
            app_pgrp = app.pid
            log_reader = AppLogReader(
                app.stdout, run_root / "application.log", started
            )
            log_reader.start()
            deadline = time.monotonic() + args.timeout_seconds
            next_sample = time.monotonic()
            next_ui_probe = time.monotonic()

            while app.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    timed_out = True
                    terminate_group(app_pgrp)
                    break
                if ui and browser is None and now >= next_ui_probe:
                    next_ui_probe = now + 0.05
                    state = health_status(case_port)
                    if state == "paused":
                        ui_paused_t_s = time.monotonic() - started
                        browser_log_stream = (run_root / "chromium.log").open(
                            "w", encoding="utf-8"
                        )
                        browser = subprocess.Popen(
                            browser_command(profile, case_port),
                            cwd=PROJECT_ROOT,
                            stdout=browser_log_stream,
                            stderr=subprocess.STDOUT,
                            text=True,
                            start_new_session=True,
                        )
                        browser_pgrp = browser.pid
                        browser_started_t_s = time.monotonic() - started
                if (
                    ui
                    and browser is not None
                    and ui_resumed_t_s is None
                    and log_reader.ui_state_request_seen.is_set()
                ):
                    if not post_control(case_port, "resume"):
                        raise RuntimeError("Could not resume runtime after UI launch")
                    ui_resumed_t_s = time.monotonic() - started
                if (
                    ui
                    and browser_started_t_s is not None
                    and ui_resumed_t_s is None
                    and now - started - browser_started_t_s > 20.0
                ):
                    raise RuntimeError("Visible Chromium did not load the UI within 20s")
                if (
                    ui
                    and browser is not None
                    and browser.poll() is not None
                    and log_reader.end_t_s is None
                ):
                    browser_exited_early = True
                    raise RuntimeError(
                        "Visible Chromium exited before media processing completed"
                    )
                if now >= next_sample:
                    phase = "loading"
                    if log_reader.first_frame_t_s is not None:
                        phase = "inference"
                    if log_reader.end_t_s is not None:
                        phase = "completed"
                    write_resource_row(
                        writer, sampler.sample(phase, app_pgrp, browser_pgrp)
                    )
                    stream.flush()
                    next_sample += args.sample_interval
                time.sleep(min(0.02, max(0.0, next_sample - time.monotonic())))

            app_returncode = app.wait(timeout=5)
            browser_returncode_at_app_exit = browser.poll() if browser else None
            write_resource_row(
                writer, sampler.sample("completed", app_pgrp, browser_pgrp)
            )
            stream.flush()
    except Exception as error:
        case_error = error
    finally:
        if app is not None and app.poll() is None:
            terminate_group(app.pid)
            try:
                app_returncode = app.wait(timeout=5)
            except subprocess.TimeoutExpired:
                app_returncode = None
        terminate_group(browser_pgrp)
        if browser is not None:
            try:
                browser.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if browser_log_stream is not None:
            browser_log_stream.close()
        if log_reader is not None:
            log_reader.join(timeout=5)
        # The directory is created above and never accepts a user-provided path.
        shutil.rmtree(profile, ignore_errors=True)

    session_root = sessions_root / run_id
    summary_path = session_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    completed_at = datetime.now().astimezone().isoformat(timespec="milliseconds")
    metadata = {
        "schema": "insight_cup_offline_benchmark_run.v1",
        "run_id": run_id,
        "status": "completed"
        if (
            case_error is None
            and app_returncode == 0
            and not timed_out
            and summary.get("status") == "completed"
            and summary.get("processed_frames") == expected_frames
            and log_reader is not None
            and log_reader.frame_records == expected_frames
            and log_reader.end_t_s is not None
            and (
                not ui
                or (
                    browser_started_t_s is not None
                    and ui_paused_t_s is not None
                    and ui_resumed_t_s is not None
                    and log_reader is not None
                    and log_reader.first_frame_t_s is not None
                    and log_reader.first_frame_t_s >= ui_resumed_t_s
                    and log_reader.ui_stream_request_seen.is_set()
                    and not browser_exited_early
                    and browser_returncode_at_app_exit is None
                )
            )
        )
        else "failed",
        "video_key": spec.key,
        "video": str(video),
        "video_sha256": spec.expected_sha256,
        "expected_frames": expected_frames,
        "expected_fps": spec.expected_fps,
        "ui": ui,
        "visible_browser": ui,
        "command": command,
        "started_at": started_wall,
        "completed_at": completed_at,
        "wall_seconds": round(time.monotonic() - started, 6),
        "first_frame_t_s": log_reader.first_frame_t_s if log_reader else None,
        "last_frame_t_s": log_reader.last_frame_t_s if log_reader else None,
        "runtime_end_t_s": log_reader.end_t_s if log_reader else None,
        "frame_log_records": log_reader.frame_records if log_reader else 0,
        "ui_stream_request_seen": (
            log_reader.ui_stream_request_seen.is_set() if log_reader else False
        ),
        "browser_started_t_s": browser_started_t_s,
        "ui_paused_t_s": ui_paused_t_s,
        "ui_resumed_t_s": ui_resumed_t_s,
        "browser_returncode_at_app_exit": browser_returncode_at_app_exit,
        "browser_exited_early": browser_exited_early,
        "app_returncode": app_returncode,
        "timed_out": timed_out,
        "error": str(case_error) if case_error is not None else None,
        "processed_frames": summary.get("processed_frames"),
        "session_status": summary.get("status"),
        "session_root": str(session_root),
        "resources_csv": str(resources_path),
        "application_log": str(run_root / "application.log"),
        "rknn_performance_jsonl": str(run_root / "rknn_perf.jsonl"),
        "npu_debug_load_readable": os.access(NPU_DEBUG_LOAD, os.R_OK),
        "logical_cpus": LOGICAL_CPUS,
        "sample_interval_s": args.sample_interval,
        "ui_port": case_port if ui else None,
    }
    atomic_json(meta_path, metadata)
    if metadata["status"] != "completed":
        raise RuntimeError(
            f"{run_id} failed: app_rc={app_returncode}, "
            f"session_status={summary.get('status')}, timed_out={timed_out}, "
            f"error={case_error}"
        )
    return metadata


def selected_runs(only: Iterable[str] | None) -> list[tuple[str, bool]]:
    if not only:
        return list(DEFAULT_RUN_ORDER)
    selected = set(only)
    return [
        (video, ui)
        for video, ui in DEFAULT_RUN_ORDER
        if f"{video}-{'ui' if ui else 'no-ui'}" in selected
    ]


def build(binary: Path, result_root: Path) -> None:
    command = [
        "cmake",
        "--build",
        str(binary.resolve().parent),
        "--target",
        binary.name,
        "--parallel",
        "2",
    ]
    with (result_root / "build.log").open("w", encoding="utf-8") as output:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Build failed; see {result_root / 'build.log'}")


def main() -> int:
    args = parse_args()
    if args.sample_interval <= 0 or args.baseline_seconds < 0:
        raise SystemExit("sample interval must be positive and baseline non-negative")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("max frames must be positive")
    result_root = args.output_root.resolve()
    if result_root.exists() and any(result_root.iterdir()):
        raise SystemExit(f"Refusing to reuse non-empty output root: {result_root}")
    result_root.mkdir(parents=True, exist_ok=True)

    input_metadata: dict[str, Any] = {}
    for key, spec in VIDEOS.items():
        path = args.input_root.resolve() / spec.filename
        if not path.is_file():
            raise SystemExit(f"Missing benchmark video: {path}")
        digest = spec.expected_sha256 if args.skip_hash_check else sha256_file(path)
        if not args.skip_hash_check and digest != spec.expected_sha256:
            raise SystemExit(
                f"Input hash mismatch for {path.name}: {digest} != {spec.expected_sha256}"
            )
        input_metadata[key] = {
            "path": str(path),
            "sha256": digest,
            "expected_frames": spec.expected_frames,
            "fps": spec.expected_fps,
        }

    if not args.skip_build:
        build(args.binary, result_root)
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        raise SystemExit(f"Benchmark binary is not executable: {args.binary}")

    artifact_paths = {
        "binary": args.binary.resolve(),
        "benchmark_runner": Path(__file__).resolve(),
        "benchmark_analyzer": PROJECT_ROOT
        / "tools/benchmark/analyze_offline_ui_benchmark.py",
        "yolo_rknn": PROJECT_ROOT / "models/rknn/rk3588_fp16/yolo_best.rknn",
        "face_detector_rknn": PROJECT_ROOT
        / "models/rknn/rk3588_fp16/scrfd_det_10g.rknn",
        "face_recognizer_rknn": PROJECT_ROOT
        / "models/rknn/rk3588_fp16/arcface_w600k_r50.rknn",
        "knife_rknn": PROJECT_ROOT
        / "models/rknn/rk3588_fp16/knife_pplcnetv2.rknn",
        "face_gallery": PROJECT_ROOT
        / "data/face_gallery/cpp/trainv5_gallery_r50.icg",
        "knife_sparse_manifest": PROJECT_ROOT
        / "data/knife_dataset/trainv5_grouped/manifest.json",
        "face_sparse_manifest": PROJECT_ROOT
        / "data/face_gallery/references_trainv5/manifest.json",
    }
    missing_artifacts = [str(path) for path in artifact_paths.values() if not path.is_file()]
    if missing_artifacts:
        raise SystemExit(f"Missing benchmark artifact(s): {', '.join(missing_artifacts)}")

    runs = selected_runs(args.only)
    benchmark = {
        "schema": "insight_cup_offline_benchmark.v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "project_root": str(PROJECT_ROOT),
        "binary": str(args.binary.resolve()),
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in artifact_paths.items()
        },
        "inputs": input_metadata,
        "run_order": [f"{key}-{'ui' if ui else 'no-ui'}" for key, ui in runs],
        "protocol": {
            "provider": "rknn",
            "paced_video": True,
            "video_stride": 1,
            "stage2_workers": 1,
            "shadow_log": True,
            "save_video": False,
            "save_crops": False,
            "visible_chromium_for_ui": True,
            "scheduling_priority": "normal",
            "sample_interval_s": args.sample_interval,
            "baseline_seconds": args.baseline_seconds,
            "max_frames": args.max_frames,
        },
        "host": {
            "logical_cpus": LOGICAL_CPUS,
            "npu_debug_load": str(NPU_DEBUG_LOAD),
            "npu_debug_load_readable": os.access(NPU_DEBUG_LOAD, os.R_OK),
            "npu_devfreq_load_known_invalid": True,
        },
        "runs": [],
    }
    atomic_json(result_root / "benchmark.json", benchmark)

    print(f"Benchmark output: {result_root}", flush=True)
    try:
        for index, (key, ui) in enumerate(runs, start=1):
            run_id = f"{key}-{'ui' if ui else 'no-ui'}"
            print(f"[{index}/{len(runs)}] Starting {run_id}", flush=True)
            metadata = run_case(VIDEOS[key], ui, args, result_root)
            benchmark["runs"].append(metadata)
            atomic_json(result_root / "benchmark.json", benchmark)
            print(
                f"[{index}/{len(runs)}] Completed {run_id}: "
                f"{metadata['processed_frames']} frames",
                flush=True,
            )
    except KeyboardInterrupt:
        benchmark["interrupted"] = True
        atomic_json(result_root / "benchmark.json", benchmark)
        return 130
    except Exception as error:
        benchmark["error"] = str(error)
        atomic_json(result_root / "benchmark.json", benchmark)
        print(f"Benchmark failed: {error}", file=sys.stderr)
        return 1
    benchmark["completed_at"] = datetime.now().astimezone().isoformat(
        timespec="milliseconds"
    )
    atomic_json(result_root / "benchmark.json", benchmark)
    print("All selected runs completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
