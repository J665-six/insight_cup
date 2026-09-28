#!/usr/bin/env python3
"""Install a detection-and-recognition-only subset of InsightFace buffalo_l."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from insight_cup.face.engine import DEFAULT_MODEL_NAME, DEFAULT_MODEL_ROOT


MODEL_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)
DEFAULT_DOWNLOAD_URL = f"https://gh-proxy.com/{MODEL_URL}"
OFFICIAL_ARCHIVE_MD5 = "6c0e929fd3b6ab517170b732ced18c68"
REQUIRED_FILES = frozenset({"det_10g.onnx", "w600k_r50.onnx"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download buffalo_l but retain only SCRFD-10G detection and "
            "ArcFace-R50 recognition models."
        )
    )
    parser.add_argument("--root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--download-url",
        default=DEFAULT_DOWNLOAD_URL,
        help="Transport URL; content is checked against the official archive MD5.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    model_dir = root / "models" / args.name
    existing = {path.name for path in model_dir.glob("*.onnx")} if model_dir.exists() else set()
    if existing == REQUIRED_FILES:
        print(f"Face-only model pack already installed: {model_dir}")
        return
    if existing:
        raise RuntimeError(
            f"Model directory already contains unexpected ONNX files: {sorted(existing)}"
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="insightface-model-") as temp_value:
        archive_path = Path(temp_value) / "buffalo_l.zip"
        print(f"Downloading official model pack via: {args.download_url}", flush=True)
        urllib.request.urlretrieve(args.download_url, archive_path)
        archive_md5 = md5(archive_path)
        if archive_md5 != OFFICIAL_ARCHIVE_MD5:
            raise RuntimeError(
                "Downloaded archive checksum mismatch: "
                f"expected {OFFICIAL_ARCHIVE_MD5}, got {archive_md5}"
            )
        print("Keeping only detection and recognition weights", flush=True)
        with zipfile.ZipFile(archive_path) as archive:
            members = {
                Path(info.filename).name: info
                for info in archive.infolist()
                if not info.is_dir() and Path(info.filename).name in REQUIRED_FILES
            }
            missing = REQUIRED_FILES - set(members)
            if missing:
                raise RuntimeError(f"Model archive is missing: {sorted(missing)}")
            for filename in sorted(REQUIRED_FILES):
                destination = model_dir / filename
                partial = destination.with_suffix(destination.suffix + ".part")
                with archive.open(members[filename]) as source, partial.open("wb") as output:
                    shutil.copyfileobj(source, output)
                partial.replace(destination)

    installed_files = []
    for filename in sorted(REQUIRED_FILES):
        path = model_dir / filename
        installed_files.append(
            {
                "name": filename,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema": "insightface_face_only_models.v1",
        "installed_at": datetime.now().isoformat(timespec="seconds"),
        "source": MODEL_URL,
        "transport_url": args.download_url,
        "source_archive_md5": OFFICIAL_ARCHIVE_MD5,
        "model_dir": str(model_dir),
        "included_tasks": ["detection", "recognition"],
        "excluded_tasks": [
            "3d_landmark",
            "face_reconstruction",
            "liveness",
            "attribute",
            "face_swap",
        ],
        "files": installed_files,
        "license_note": (
            "InsightFace pretrained models are licensed for non-commercial "
            "research use only by their provider."
        ),
    }
    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Installed face-only model pack: {model_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
