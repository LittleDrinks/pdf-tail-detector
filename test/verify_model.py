#!/usr/bin/env python3
"""Compare official Paddle inference with ONNX on untouched real page images."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import paddle
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paddle.inference import Config, create_predictor

from src.layout import PPDocLayout, model_inputs, render_page, sha256

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model-dir", type=Path, default=Path("outputs/models/PP-DocLayout-M"))
parser.add_argument("--output", type=Path, default=Path("outputs/verification/model-consistency.json"))
args = parser.parse_args()
converted = PPDocLayout(args.model_dir, 2)
for name in ("inference.json", "inference.pdiparams"):
    if sha256(args.model_dir / name) != converted.manifest["sha256"][name]:
        raise ValueError(f"Native weight checksum mismatch: {name}")
config = Config(str(args.model_dir / "inference.json"), str(args.model_dir / "inference.pdiparams"))
config.disable_gpu()
# Paddle 3.3's oneDNN PIR pass rejects this model's scale attributes.
config.disable_mkldnn()
config.set_cpu_math_library_num_threads(2)
config.disable_glog_info()
reference = create_predictor(config)


def paddle_raw(rgb):
    inputs = model_inputs(rgb)
    for key in reference.get_input_names():
        handle = reference.get_input_handle(key)
        handle.reshape(inputs[key].shape)
        handle.copy_from_cpu(inputs[key])
    reference.run()
    outputs = [reference.get_output_handle(key).copy_to_cpu() for key in reference.get_output_names()]
    return next(output for output in outputs if output.ndim == 2 and output.shape[1] == 6)


checks = []
for path, page_number in [("test/fixtures/iclr/what-does-automatic-differentiation-compute.pdf", 7),
                          ("test/fixtures/iclr/efficiently-computing-similarities.pdf", 3),
                          ("test/fixtures/iclr/efficiently-computing-similarities.pdf", 9)]:
    with pymupdf.open(path) as pdf:
        rgb, _ = render_page(pdf[page_number - 1], 150)
    tick = time.perf_counter()
    native = paddle_raw(rgb)
    native_time = time.perf_counter() - tick
    tick = time.perf_counter()
    onnx = converted.raw(rgb)
    onnx_time = time.perf_counter() - tick
    # Match class and nearest coordinates, not unstable NMS row order.
    active = native[native[:, 1] >= .5]
    available = [i for i, row in enumerate(onnx) if row[1] >= .5]
    pairs = []
    for row in active:
        candidates = [i for i in available if onnx[i, 0] == row[0]]
        if not candidates:
            raise AssertionError(f"Missing class {row[0]}: {path}:{page_number}")
        index = min(candidates, key=lambda i: np.max(np.abs(onnx[i, 2:] - row[2:])))
        available.remove(index)
        pairs.append({"class": int(row[0]), "score_delta": float(abs(row[1] - onnx[index, 1])),
                      "max_coordinate_delta_pixels": float(np.max(np.abs(row[2:] - onnx[index, 2:])))})
    passed = not available and all(p["score_delta"] < .001 and p["max_coordinate_delta_pixels"] < .2 for p in pairs)
    checks.append({"pdf": path, "page": page_number, "boxes": len(active), "passed": passed,
                   "paddle_seconds": native_time, "onnx_seconds": onnx_time,
                   "matches": pairs, "native_raw": native.tolist(), "onnx_raw": onnx.tolist()})
    print(f"{path}:{page_number}: {passed}, {len(active)} boxes", flush=True)
    if not passed:
        raise AssertionError("Native / ONNX consistency failed")
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({"passed": True, "thresholds": {"score_delta": .001, "coordinate_delta_pixels": .2},
                                  "reference": {**converted.metadata(), "backend": "paddle", "runtime_version": paddle.__version__}, "onnx": converted.metadata(), "checks": checks}, indent=2) + "\n")
