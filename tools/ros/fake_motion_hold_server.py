#!/usr/bin/env python3
"""Test-only replacement for /navigation_mission/set_motion_hold.

This node never talks to a motor controller. It records each request, keeps
the active hold sources in memory, and returns the state that a navigation
mission manager would expose to the inspection pipeline.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from typing import Set

import rclpy
from mission_control_interfaces.srv import SetMotionHold
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


class FakeMotionHoldServer(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("fake_motion_hold_server")
        self._lock = threading.Lock()
        self._active_sources: Set[str] = set()
        self._request_id = 0
        self._delay_s = max(0.0, args.delay_ms / 1000.0)
        self._reject_hold = args.reject_hold
        self._reject_release = args.reject_release
        self._json_log = args.json_log
        self._service = self.create_service(
            SetMotionHold,
            args.service_name,
            self._handle_request,
        )
        self.get_logger().info(
            "FAKE motion hold server ready service=%s delay_ms=%d "
            "reject_hold=%s reject_release=%s"
            % (
                args.service_name,
                args.delay_ms,
                str(args.reject_hold).lower(),
                str(args.reject_release).lower(),
            )
        )

    def _handle_request(
        self,
        request: SetMotionHold.Request,
        response: SetMotionHold.Response,
    ) -> SetMotionHold.Response:
        if self._delay_s:
            time.sleep(self._delay_s)

        with self._lock:
            self._request_id += 1
            request_id = self._request_id
            source = request.source.strip() or "<empty>"
            rejected = self._reject_hold if request.hold else self._reject_release
            if not rejected:
                if request.hold:
                    self._active_sources.add(source)
                else:
                    self._active_sources.discard(source)
            active = sorted(self._active_sources)
            motion_held = bool(active)

        response.success = not rejected
        response.motion_held = motion_held
        response.active_sources = active
        response.message = (
            "fake service rejected request"
            if rejected
            else ("motion held" if motion_held else "motion released")
        )
        payload = {
            "request_id": request_id,
            "source": request.source,
            "hold": request.hold,
            "reason": request.reason,
            "success": response.success,
            "motion_held": response.motion_held,
            "active_sources": active,
            "message": response.message,
        }
        if self._json_log:
            print("FAKE_HOLD " + json.dumps(payload, ensure_ascii=False), flush=True)
        else:
            self.get_logger().info(
                "FAKE_HOLD request_id=%d source=%s hold=%s reason=%s "
                "success=%s motion_held=%s active_sources=%s"
                % (
                    request_id,
                    request.source or "<empty>",
                    str(request.hold).lower(),
                    request.reason or "<empty>",
                    str(response.success).lower(),
                    str(response.motion_held).lower(),
                    ",".join(active) if active else "<none>",
                )
            )
        return response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fake SetMotionHold server for local insight_cup testing"
    )
    parser.add_argument(
        "--service-name",
        default="/navigation_mission/set_motion_hold",
        help="ROS 2 service name",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=0,
        help="artificial response delay in milliseconds",
    )
    parser.add_argument(
        "--reject-hold",
        action="store_true",
        help="return success=false for hold=true requests",
    )
    parser.add_argument(
        "--reject-release",
        action="store_true",
        help="return success=false for hold=false requests",
    )
    parser.add_argument(
        "--json-log",
        action="store_true",
        help="print each request as one machine-readable JSON line",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = FakeMotionHoldServer(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
