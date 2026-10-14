#!/usr/bin/env python3
"""正式识别 /insight -> 独立串口播报，绕过导航总控。

识别由正式 C++ 程序 ./start.sh 完整执行，包括 ROI、面积门控、停车确认、
二阶段识别和正式稳定判断。本脚本只接收最终 /insight，按导航总控同样的
目标顺序、去重、复位和文本长度条件决定播报，不再运行第二套识别。

先在其他终端启动 fake_motion_hold_server.py 和正式 ./start.sh，随后运行：
    python3 vioce.py --port /dev/ttyACM0
所有终端先 source /opt/ros/humble/setup.bash 及导航工作区 install/setup.bash。
--text 单测播报；--dry-run 接收正式结果但只打印，不打开串口。
只由本脚本占用串口；无需启动导航总控或 cup_car_serial。
DM2325 协议：TTS, + GBK 文本 + CRLF，经 UART7 播报，无播放完成 ACK。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
import threading
import time
from collections import deque


# DM2325 (1a6cbf5): 96-byte command buffer, minus NUL and the "TTS," prefix.
MAX_GBK_BYTES = 91


def target_announcement(target):
    """Match navigation_mission_manager/src/environment.cpp exactly."""
    if not isinstance(target, str) or re.fullmatch(r"[fk][0-9]+", target) is None:
        return ""
    return f"发现{target[1:]}号{'嫌疑人' if target[0] == 'f' else '刀具'}"


def tts_frames(text):
    """Split on phrase boundaries when possible; never split a GBK character."""
    text = text.strip()
    if not text or any(character in text for character in "\r\n\0"):
        raise ValueError("播报文本不能为空，也不能含换行或 NUL")
    text.encode("gbk")  # Reject unsupported characters before sending any frame.
    chunks, current = [], ""
    for phrase in re.findall(r"[^，。；]+[，。；]?|[，。；]", text):
        if current and len((current + phrase).encode("gbk")) > MAX_GBK_BYTES:
            chunks.append(current)
            current = ""
        for character in phrase:
            if len((current + character).encode("gbk")) > MAX_GBK_BYTES:
                chunks.append(current)
                current = ""
            current += character
    if current:
        chunks.append(current)
    return [(chunk, b"TTS," + chunk.encode("gbk") + b"\r\n") for chunk in chunks]


class InsightAnnouncements:
    """Mirror MissionManager::on_insight and enqueue_speech_text conditions.

    Formal /insight already passed the recognition gates; never add another
    stable-frame counter or a timed rearm. Empty/no-target messages rearm the
    announcements exactly as the navigation manager does.
    """

    def __init__(self, speaker):
        self.speaker = speaker
        self.announced = set()

    def receive(self, payload):
        try:
            event = json.loads(payload)
        except (json.JSONDecodeError, TypeError) as error:
            print(f"INSIGHT INVALID JSON: {error}", file=sys.stderr, flush=True)
            return
        if not isinstance(event, dict) or not isinstance(event.get("targets", []), list):
            print("INSIGHT INVALID: expected JSON object with targets array", file=sys.stderr, flush=True)
            return
        # Keep the original order, as MissionManager::json_targets does.
        targets = list(dict.fromkeys(
            value for value in event.get("targets", []) if isinstance(value, str) and value))
        detected = event.get("target_detected")
        if not isinstance(detected, bool):
            detected = bool(targets)
        if not detected or not targets:
            self.announced.clear()
            return
        fresh, phrases = [], []
        for target in targets:
            phrase = target_announcement(target)
            if not phrase or target in self.announced:
                continue
            self.announced.add(target)
            fresh.append(target)
            phrases.append(phrase)
        if not phrases:
            return
        text = "，".join(phrases).strip(" \t")
        # Navigation rejects UTF-8 strings over 128 bytes before queueing.
        if len(text.encode("utf-8")) > 128 or "\n" in text or "\r" in text:
            print("INSIGHT SKIPPED: exceeds formal 128-byte speech limit", flush=True)
            return
        print(f"INSIGHT EVENT targets={json.dumps(fresh)} text={text}", flush=True)
        self.speaker.enqueue(text)


def serial_device(port):
    if port != "auto":
        return port
    candidates = sorted(Path("/dev/serial/by-id").glob("*"))
    if not candidates:
        candidates = sorted(Path("/dev").glob("ttyACM*")) + sorted(Path("/dev").glob("ttyUSB*"))
    # Two stable aliases can point at the same device.
    devices = {str(path.resolve()): str(path) for path in candidates}
    if len(devices) != 1:
        detail = "、".join(devices.values()) or "未找到 USB 串口"
        raise RuntimeError(f"无法自动选择下位机串口：{detail}；请用 --port 指定")
    return next(iter(devices.values()))


class SerialSpeaker:
    """Send paced TTS frames; drain MCU telemetry independently of inference."""

    def __init__(self, args):
        self.args = args
        self.port = None
        self.thread = None
        self.stop = threading.Event()
        self.rx_error = None
        self.pending = deque()
        self.next_send = 0.0
        self.sent = 0

    def __enter__(self):
        if self.args.dry_run:
            print("DRY RUN：只打印 TTS，不打开串口", flush=True)
            return self
        try:
            import serial
        except ImportError as error:
            raise RuntimeError("缺少 pyserial：python3 -m pip install pyserial") from error
        device = serial_device(self.args.port)
        self.port = serial.Serial(
            device, self.args.baud_rate, timeout=0.1, write_timeout=2.0, exclusive=True,
        )
        print(f"SERIAL OPEN {device} {self.args.baud_rate} 8N1", flush=True)
        # Wait for USB startup and terminate any stale partial command. These
        # steps match the sequence observed to work on the DM2325 board; the
        # firmware provides no ACK to identify which step resolved the issue.
        try:
            time.sleep(self.args.serial_settle)
            self.port.reset_input_buffer()
            if self.port.write(b"\r\n") != 2:
                raise RuntimeError("串口命令同步发送不完整")
            self.port.flush()
            time.sleep(0.1)
            self.thread = threading.Thread(target=self._read_rx, daemon=True)
            self.thread.start()
        except BaseException:
            self.port.close()
            raise
        return self

    def _read_rx(self):
        buffer = bytearray()
        first = True
        try:
            while not self.stop.is_set():
                buffer.extend(self.port.read(min(4096, self.port.in_waiting or 1)))
                while b"\n" in buffer:
                    line, _, rest = buffer.partition(b"\n")
                    buffer = bytearray(rest)
                    line = line.rstrip(b"\r")
                    if line and (first or self.args.show_rx):
                        print(f"MCU RX {line.decode('ascii', errors='replace')}", flush=True)
                        first = False
                if len(buffer) > 4096:
                    buffer.clear()
        except Exception as error:
            if not self.stop.is_set():
                self.rx_error = error

    def enqueue(self, text):
        frames = tts_frames(text)
        if len(self.pending) + len(frames) > 64:
            raise RuntimeError("播报积压超过 64 条，请增大发送间隔或减少目标")
        self.pending.extend(frames)
        print(f"QUEUED {text}", flush=True)

    def tick(self):
        if self.rx_error is not None:
            raise RuntimeError(f"下位机串口读取失败：{self.rx_error}") from self.rx_error
        if not self.pending or time.monotonic() < self.next_send:
            return
        text, frame = self.pending[0]
        if self.port is not None:
            count = self.port.write(frame)
            if count != len(frame):
                raise RuntimeError(f"TTS 发送不完整：{count}/{len(frame)} 字节")
            self.port.flush()
        self.pending.popleft()
        self.sent += 1
        print(f"{'DRY TTS' if self.args.dry_run else 'TTS SENT'} #{self.sent} {text} | {frame.hex(' ')}", flush=True)
        # Firmware exposes no playback ACK. Allow time for the whole phrase.
        self.next_send = time.monotonic() + max(self.args.speech_gap, len(text) * 0.30 + 0.8)

    def __exit__(self, *unused):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
        if self.port is not None:
            self.port.close()
        if self.pending:
            print(f"退出：还有 {len(self.pending)} 条播报未发送", flush=True)


def run_insight(args, speaker):
    try:
        import rclpy
        from rclpy.executors import ExternalShutdownException
        from rclpy.node import Node
        from std_msgs.msg import String
    except ImportError as error:
        raise RuntimeError("请先 source /opt/ros/humble/setup.bash，再加载导航工作区 install/setup.bash") from error

    rclpy.init(args=[])
    node = None
    try:
        node = Node("insight_voice_test")
        announcements = InsightAnnouncements(speaker)
        subscription = node.create_subscription(
            String, args.insight_topic, lambda message: announcements.receive(message.data), 10)
        node.get_logger().info(
            f"READY listening {args.insight_topic} -> MCU TTS; "
            "recognition/ROI/hold/stability are owned by the formal ./start.sh runtime")
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.02)
            speaker.tick()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv=None):
    p = argparse.ArgumentParser(
        description="正式 /insight -> 下位机播报测试，绕过导航总控；另行启动正式识别和假停车服务")
    p.add_argument("--port", default="auto", help="下位机串口，如 /dev/ttyACM0；默认自动选唯一 USB 串口")
    p.add_argument("--baud-rate", type=int, default=115200)
    p.add_argument("--insight-topic", default="/insight", help="正式综合结果话题，默认 /insight")
    p.add_argument("--speech-gap", type=float, default=3.0, help="两条 TTS 的最短间隔秒数，长文本自动延长")
    p.add_argument("--serial-settle", type=float, default=0.8, help="打开 USB 串口后等待下位机稳定的秒数，默认 0.8")
    p.add_argument("--text", help="发送指定文本后退出，不加载模型或相机")
    p.add_argument("--dry-run", action="store_true", help="只打印 TTS，不打开串口")
    p.add_argument("--show-rx", action="store_true", help="打印所有下位机遥测；默认只打印第一条")
    p.add_argument("--list-ports", action="store_true", help="列出串口后退出")
    args = p.parse_args(argv)
    if args.list_ports:
        try:
            from serial.tools import list_ports
        except ImportError:
            p.error("缺少 pyserial：python3 -m pip install pyserial")
        ports = [port for port in list_ports.comports()
                 if port.device.startswith(("/dev/ttyACM", "/dev/ttyUSB"))]
        for port in ports:
            print(f"{port.device}\t{port.description}\t{port.hwid}")
        if not ports:
            print("未找到串口，请连接下位机 USB")
        return 0
    if args.baud_rate < 1:
        p.error("波特率必须为正整数")
    for option in ("speech_gap", "serial_settle"):
        if not math.isfinite(getattr(args, option)) or getattr(args, option) <= 0:
            p.error(f"--{option.replace('_', '-')} 必须是有限正数")
    if args.text is not None:
        try:
            tts_frames(args.text)  # Validate before opening the port.
        except (ValueError, UnicodeError) as error:
            p.error(str(error))
    try:
        with SerialSpeaker(args) as speaker:
            if args.text is not None:
                speaker.enqueue(args.text)
                while speaker.pending:
                    speaker.tick()
                    time.sleep(0.02)
                # Keep USB connected through the observed working window and
                # estimated speech duration. This is not a playback ACK.
                if not args.dry_run:
                    deadline = max(speaker.next_send, time.monotonic() + 4.0)
                    print("等待播报，保持串口连接…", flush=True)
                    while time.monotonic() < deadline:
                        speaker.tick()
                        time.sleep(0.05)
            else:
                run_insight(args, speaker)
    except KeyboardInterrupt:
        print("\n已停止", flush=True)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
