"""Dependency-free HTTP server for the live debug interface."""

from __future__ import annotations

import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .state import RuntimeState


LOGGER = logging.getLogger("recognition.web")
UI_ROOT = Path(__file__).resolve().parent / "ui"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
MJPEG_BOUNDARY = "insightcupframe"


class DebugHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], state: RuntimeState) -> None:
        super().__init__(address, DebugRequestHandler)
        self.runtime_state = state


class DebugRequestHandler(BaseHTTPRequestHandler):
    server: DebugHTTPServer
    protocol_version = "HTTP/1.1"

    def do_HEAD(self) -> None:
        self._head_only = True
        try:
            self.do_GET()
        finally:
            self._head_only = False

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in STATIC_FILES:
            filename, content_type = STATIC_FILES[parsed.path]
            self._send_bytes(
                HTTPStatus.OK,
                (UI_ROOT / filename).read_bytes(),
                content_type,
                cache="no-cache",
            )
            return
        if parsed.path == "/api/state":
            self._send_json(HTTPStatus.OK, self.server.runtime_state.snapshot())
            return
        if parsed.path == "/api/events":
            query = parse_qs(parsed.query)
            try:
                limit = int(query.get("limit", ["50"])[0])
            except ValueError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid limit"})
                return
            self._send_json(
                HTTPStatus.OK,
                {"events": self.server.runtime_state.recent_events(limit)},
            )
            return
        if parsed.path == "/api/frame.jpg":
            frame = self.server.runtime_state.latest_frame()
            if frame is None:
                self._send_bytes(HTTPStatus.NO_CONTENT, b"", "image/jpeg")
            else:
                self._send_bytes(
                    HTTPStatus.OK,
                    frame,
                    "image/jpeg",
                    cache="no-store, max-age=0",
                )
            return
        if parsed.path == "/api/stream.mjpg":
            self._send_mjpeg_stream()
            return
        if parsed.path == "/healthz":
            self._send_json(
                HTTPStatus.OK,
                {"ok": True, "status": self.server.runtime_state.snapshot()["status"]},
            )
            return
        if parsed.path == "/favicon.ico":
            self._send_bytes(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/control":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > 4096:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid body"})
            return
        try:
            payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
            action = str(payload["action"])
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
            return
        accepted, message = self.server.runtime_state.control(action)
        status = HTTPStatus.OK if accepted else HTTPStatus.CONFLICT
        self._send_json(
            status,
            {
                "accepted": accepted,
                "message": message,
                "state": self.server.runtime_state.snapshot(),
            },
        )

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        self._send_bytes(
            status,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            cache="no-store",
        )

    def _send_mjpeg_stream(self) -> None:
        try:
            self.send_response(HTTPStatus.OK.value)
            self.send_header(
                "Content-Type",
                f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
            )
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            if getattr(self, "_head_only", False):
                return

            version = 0
            terminal_states = {"completed", "stopped", "error"}
            while True:
                next_version, frame, status = self.server.runtime_state.wait_for_frame(
                    version,
                    timeout=1.0,
                )
                if next_version == version or frame is None:
                    if status in terminal_states:
                        return
                    continue
                version = next_version
                header = (
                    f"--{MJPEG_BOUNDARY}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(frame)}\r\n\r\n"
                ).encode("ascii")
                self.wfile.write(header)
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                if status in terminal_states:
                    return
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def _send_bytes(
        self,
        status: HTTPStatus,
        payload: bytes,
        content_type: str,
        cache: str = "no-store",
    ) -> None:
        try:
            self.send_response(status.value)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", cache)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' data:; "
                "style-src 'self'; script-src 'self'; connect-src 'self'",
            )
            self.end_headers()
            if payload and not getattr(self, "_head_only", False):
                self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            return

    def log_message(self, format_string: str, *args: object) -> None:
        LOGGER.debug("HTTP %s - %s", self.address_string(), format_string % args)


def create_debug_server(host: str, port: int, state: RuntimeState) -> DebugHTTPServer:
    return DebugHTTPServer((host, port), state)
