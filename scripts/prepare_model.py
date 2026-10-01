#!/usr/bin/env python3
"""Fetch verified official weights, optionally export an ONNX CPU model."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.layout import LABELS, MODEL_REVISION, sha256

WEIGHTS = {
    "inference.json": "5b2ec6d403904d40b21ba261ad6d1a41e12e3fab256d075c1c7362d4742a8fb7",
    "inference.pdiparams": "f374bb0269d91ab2eed393a5a2da6d73d98896ac5eeae4f9f585eba19bcbba74",
    "inference.yml": "76aeb103310f432ea773d4a6d187e15b26d92c051c5e2875598e1d49f725d70d",
}

def _download(directory: Path, name: str, expected: str) -> None:
    path = directory / name
    if path.exists() and sha256(path) == expected:
        return
    temporary = directory / (name + ".part")
    url = f"https://huggingface.co/PaddlePaddle/PP-DocLayout-M/resolve/{MODEL_REVISION}/{name}"
    print(f"Downloading {name}", flush=True)
    with urllib.request.urlopen(url, timeout=120) as source, temporary.open("wb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
    if sha256(temporary) != expected:
        temporary.unlink()
        raise ValueError(f"Official weight checksum mismatch: {name}")
    temporary.replace(path)


def prepare(directory: Path, export: bool) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    for name, expected in WEIGHTS.items():
        _download(directory, name, expected)
    manifest = {"name": "PP-DocLayout-M", "revision": MODEL_REVISION,
                "source": "https://huggingface.co/PaddlePaddle/PP-DocLayout-M",
                "labels": list(LABELS), "sha256": dict(WEIGHTS)}
    if export:
        _export(directory, manifest)
    else:
        _reuse_export(directory, manifest)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _export(directory: Path, manifest: dict) -> None:
    temporary = directory / "model.exporting.onnx"
    executable = Path(sys.executable).with_name("paddle2onnx")
    command = [str(executable), "--model_dir", str(directory), "--model_filename", "inference.json",
               "--params_filename", "inference.pdiparams", "--save_file", str(temporary),
               "--opset_version", "17", "--optimize_tool", "None"]
    print("Exporting with paddle2onnx", flush=True)
    subprocess.run(command, check=True)
    temporary.replace(directory / "model.onnx")
    manifest["sha256"]["model.onnx"] = sha256(directory / "model.onnx")
    import importlib.metadata
    manifest["conversion"] = {"paddlepaddle": importlib.metadata.version("paddlepaddle"),
                              "paddle2onnx": importlib.metadata.version("paddle2onnx"), "opset": 17}


def _reuse_export(directory: Path, manifest: dict) -> None:
    if not (directory / "model.onnx").exists():
        return
    # Do not adopt an arbitrary preexisting ONNX file without a verified manifest.
    old = json.loads((directory / "manifest.json").read_text()) if (directory / "manifest.json").exists() else {}
    expected = old.get("sha256", {}).get("model.onnx")
    if expected and sha256(directory / "model.onnx") == expected:
        manifest["sha256"]["model.onnx"] = expected
        manifest["conversion"] = old.get("conversion")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/models/PP-DocLayout-M"))
    parser.add_argument("--export", action="store_true", help="requires requirements/model.txt")
    args = parser.parse_args()
    prepare(args.model_dir, args.export)
    print(f"Model ready: {args.model_dir}")
