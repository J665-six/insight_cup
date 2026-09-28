#!/usr/bin/env python3
"""Small browser UI for testing the latest YOLO model."""

from __future__ import annotations

import argparse
import base64
import cgi
import html
import io
import json
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


DEFAULT_WEIGHTS = Path(
    "/home/j/trainv5/runs/yolo11n_trainv5_continue_latest-2/weights/best.pt"
)


def names_map(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names or [])}


def detections_from_result(result: Any, names: dict[int, str]) -> list[dict[str, Any]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    records = []
    for xyxy, conf, cls_id in zip(
        boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()
    ):
        cls_int = int(cls_id)
        records.append(
            {
                "class": names.get(cls_int, str(cls_int)),
                "confidence": round(float(conf), 4),
                "xyxy": [round(float(v), 1) for v in xyxy],
            }
        )
    return records


def render_page(body: str = "") -> bytes:
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>YOLO Test</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #18202b;
      --muted: #657080;
      --line: #d8dde5;
      --panel: #f6f8fb;
      --accent: #0c7a6d;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: #ffffff;
    }}
    header {{
      border-bottom: 1px solid var(--line);
      padding: 18px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 24px auto 48px;
    }}
    form {{
      display: grid;
      grid-template-columns: minmax(220px, 1fr) 120px 120px auto;
      gap: 12px;
      align-items: end;
      padding: 16px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    label {{
      display: grid;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
    }}
    input {{
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 8px;
      font: inherit;
    }}
    button {{
      min-height: 38px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: #fff;
      padding: 0 18px;
      font-weight: 650;
      cursor: pointer;
    }}
    .result {{
      margin-top: 22px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 20px;
      align-items: start;
    }}
    .image-wrap {{
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #f1f3f6;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: var(--panel); color: var(--muted); font-weight: 650; }}
    tr:last-child td {{ border-bottom: 0; }}
    .empty {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 14px;
    }}
    @media (max-width: 760px) {{
      header {{ padding: 14px 16px; }}
      form {{ grid-template-columns: 1fr; }}
      .result {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>YOLO Test</h1>
    <span>trainv5_continue_latest-2</span>
  </header>
  <main>
    <form method="post" action="/predict" enctype="multipart/form-data">
      <label>图片
        <input name="image" type="file" accept="image/*" required>
      </label>
      <label>conf
        <input name="conf" type="number" min="0.01" max="1" step="0.01" value="0.25">
      </label>
      <label>iou
        <input name="iou" type="number" min="0.01" max="1" step="0.01" value="0.45">
      </label>
      <button type="submit">Run</button>
    </form>
    {body}
  </main>
</body>
</html>"""
    return page.encode("utf-8")


def render_result(image_b64: str, detections: list[dict[str, Any]]) -> str:
    if detections:
        rows = "\n".join(
            "<tr>"
            f"<td>{html.escape(d['class'])}</td>"
            f"<td>{d['confidence']:.4f}</td>"
            f"<td>{html.escape(json.dumps(d['xyxy'], ensure_ascii=False))}</td>"
            "</tr>"
            for d in detections
        )
        table = (
            "<table><thead><tr><th>class</th><th>conf</th><th>xyxy</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    else:
        table = '<div class="empty">No detections</div>'
    return (
        '<section class="result">'
        f'<div class="image-wrap"><img src="data:image/jpeg;base64,{image_b64}" alt="result"></div>'
        f"<div>{table}</div>"
        "</section>"
    )


class DemoHandler(BaseHTTPRequestHandler):
    model: YOLO
    names: dict[int, str]

    def do_GET(self) -> None:
        if self.path != "/":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(render_page())

    def do_POST(self) -> None:
        if self.path != "/predict":
            self.send_error(404)
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )
        file_item = form["image"] if "image" in form else None
        if file_item is None or not getattr(file_item, "file", None):
            self.send_error(400, "No image uploaded")
            return

        conf = float(form.getfirst("conf", "0.25"))
        iou = float(form.getfirst("iou", "0.45"))
        suffix = Path(file_item.filename or "upload.jpg").suffix or ".jpg"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_item.file.read())
            tmp_path = Path(tmp.name)

        try:
            result = self.model.predict(
                source=str(tmp_path), conf=conf, iou=iou, imgsz=640, verbose=False
            )[0]
            detections = detections_from_result(result, self.names)
            plotted = result.plot()
            ok, encoded = cv2.imencode(".jpg", plotted)
            if not ok:
                raise RuntimeError("Failed to encode result image")
            image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
            body = render_result(image_b64, detections)
            payload = render_page(body)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        finally:
            tmp_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open a local YOLO image test page.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).expanduser()
    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    DemoHandler.model = YOLO(str(weights))
    DemoHandler.names = names_map(DemoHandler.model.names)

    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Model: {weights}")
    print(f"Open: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
