"""PDF geometry and the pinned PP-DocLayout-M CPU adapter.

All rectangles use unrotated, CropBox-relative PyMuPDF coordinates: x0,y0,x1,y1.
The model identifies regions; its text boxes are never paragraph identifiers.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf

LABELS = ("paragraph_title", "image", "text", "number", "abstract", "content",
          "figure_title", "formula", "table", "table_title", "reference", "doc_title",
          "footnote", "header", "algorithm", "footer", "seal", "chart_title", "chart",
          "formula_number", "header_image", "footer_image", "aside_text")
MODEL_REVISION = "7dbfcce3154a55776dc71ca026a4a2a8388dad8d"
PREPROCESS_VERSION = "ppdoc-m-rgb-resize640-imagenet-v1"
Rect = tuple[float, float, float, float]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class Glyph:
    text: str
    bbox: Rect
    origin: tuple[float, float]
    size: float
    font: str


@dataclass
class TextLine:
    page: int
    id: str
    glyphs: list[Glyph]
    text: str
    bbox: Rect
    baseline: float
    size: float
    kind: str = "unknown"
    reason: str = "uncovered"
    region_id: str | None = None
    region_score: float | None = None
    region_label: str | None = None
    repairs: list[str] = field(default_factory=list)


@dataclass
class Region:
    id: str
    label: str
    score: float
    bbox: Rect


@dataclass
class PageData:
    number: int
    width: float
    height: float
    rotation: int
    cropbox: Rect
    lines: list[TextLine]
    regions: list[Region] = field(default_factory=list)
    status: str = "checked"
    geometry: dict = field(default_factory=dict)


def enclosing(boxes: list[Rect]) -> Rect:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def glyph_text(glyphs: list[Glyph]) -> str:
    ordered = sorted(glyphs, key=lambda g: (g.bbox[0], g.origin[1]))
    words: list[str] = []
    previous: Glyph | None = None
    for glyph in (glyph for glyph in ordered if glyph.text.strip()):
        if previous:
            gap = glyph.bbox[0] - previous.bbox[2]
            if gap > max(1.5, glyph.size * .18) and previous.text[-1:] not in "([{":
                words.append(" ")
        words.append(glyph.text)
        previous = glyph
    return "".join(words).strip()


def make_line(page: int, identifier: str, glyphs: list[Glyph]) -> TextLine:
    visible = [g for g in glyphs if g.text.strip()]
    # The largest common type size supplies the baseline, not a superscript.
    size = statistics.median(g.size for g in visible)
    ordinary = [g for g in visible if g.size >= size * .95]
    baseline = statistics.median(g.origin[1] for g in ordinary)
    return TextLine(page, identifier, visible, glyph_text(visible),
                    enclosing([g.bbox for g in visible]), baseline, size)


def _extract_line(page: int, identifier: str, line: dict) -> TextLine | None:
    # Rotated writing / vertical scripts are outside the English-paper scope.
    if abs(line["dir"][0] - 1) > .01 or abs(line["dir"][1]) > .01:
        return None
    glyphs = [Glyph(char["c"], tuple(char["bbox"]), tuple(char["origin"]),
                    float(span["size"]), span["font"])
              for span in line["spans"] for char in span.get("chars", []) if char["c"].strip()]
    return make_line(page, identifier, glyphs) if glyphs else None


def extract_page(page: pymupdf.Page) -> PageData:
    raw_lines = [(f"{bi}:{li}", line) for bi, block in enumerate(page.get_text("rawdict")["blocks"])
                 if block["type"] == 0 for li, line in enumerate(block["lines"])]
    fragments = [_extract_line(page.number + 1, identifier, line) for identifier, line in raw_lines]
    size = page.rect * page.derotation_matrix
    return PageData(page.number + 1, size.width, size.height, page.rotation,
                    tuple(page.cropbox), [line for line in fragments if line is not None])


def render_page(page: pymupdf.Page, dpi: float) -> tuple[np.ndarray, dict]:
    """Normalize rotation before rendering, then restore the caller's page."""
    rotation = page.rotation
    scale = dpi / 72
    try:
        page.set_rotation(0)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale),
                             colorspace=pymupdf.csRGB, alpha=False, annots=False)
        rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()
        geometry = {"rotation_normalized": rotation, "scale": scale,
                    "pixmap_origin": [pix.x, pix.y], "pixels": [pix.width, pix.height],
                    "coordinate_system": "unrotated-crop-relative-xyxy"}
        return rgb, geometry
    finally:
        page.set_rotation(rotation)


def map_box(box: list[float], geometry: dict) -> Rect:
    scale = geometry["scale"]
    px, py = geometry["pixmap_origin"]
    return ((box[0] + px) / scale, (box[1] + py) / scale,
            (box[2] + px) / scale, (box[3] + py) / scale)


def model_inputs(rgb: np.ndarray) -> dict[str, np.ndarray]:
    import cv2
    height, width = rgb.shape[:2]
    image = cv2.resize(rgb, (640, 640), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    image = (image - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return {"image": image.transpose(2, 0, 1)[None].copy(),
            "im_shape": np.array([[640, 640]], np.float32),
            "scale_factor": np.array([[640 / height, 640 / width]], np.float32)}


def _model_manifest(model_dir: Path) -> dict:
    manifest = json.loads((model_dir / "manifest.json").read_text())
    if manifest.get("revision") != MODEL_REVISION or manifest.get("labels") != list(LABELS):
        raise ValueError("Expected the pinned 23-class PP-DocLayout-M model.")
    expected = manifest.get("sha256", {}).get("model.onnx")
    if not expected or sha256(model_dir / "model.onnx") != expected:
        raise ValueError("Model checksum mismatch: model.onnx")
    return manifest


class PPDocLayout:
    def __init__(self, model_dir: Path, threads: int = 2):
        import onnxruntime as ort
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model missing: {model_dir}. Run scripts/prepare_model.py first.")
        self.manifest = _model_manifest(model_dir)
        self.threads, self.version = threads, ort.__version__
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"])
        self.names = [entry.name for entry in self.session.get_inputs()]

    def raw(self, rgb: np.ndarray) -> np.ndarray:
        inputs = model_inputs(rgb)
        unexpected = set(self.names) - set(inputs)
        if unexpected:
            raise ValueError(f"Unrecognized model inputs: {unexpected}")
        outputs = self.session.run(None, {key: inputs[key] for key in self.names})
        boxes = next((output for output in outputs if output.ndim == 2 and output.shape[1] == 6), None)
        if boxes is None:
            raise ValueError(f"Expected decoded [class,score,x0,y0,x1,y1], got {[output.shape for output in outputs]}")
        if not np.isfinite(boxes).all():
            raise ValueError("Model returned non-finite coordinates.")
        return boxes

    def predict(self, rgb: np.ndarray, geometry: dict, page: int, score: float = .5) -> list[Region]:
        height, width = rgb.shape[:2]
        regions = []
        for index, row in enumerate(self.raw(rgb)):
            cls, confidence, x0, y0, x1, y1 = row.tolist()
            if confidence < score or not 0 <= int(cls) < len(LABELS):
                continue
            box = [max(0, min(width, x0)), max(0, min(height, y0)),
                   max(0, min(width, x1)), max(0, min(height, y1))]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            regions.append(Region(f"p{page}-r{index}", LABELS[int(cls)], confidence, map_box(box, geometry)))
        return regions

    def metadata(self) -> dict:
        return {"name": "PP-DocLayout-M", "revision": MODEL_REVISION,
                "backend": "onnx", "runtime_version": self.version, "threads": self.threads,
                "preprocess": PREPROCESS_VERSION, "labels": list(LABELS),
                "sha256": self.manifest["sha256"]}


def coverage(line: TextLine, region: Region) -> float:
    x0, y0, x1, y1 = region.bbox
    return sum(x0 <= (g.bbox[0] + g.bbox[2]) / 2 <= x1 and
               y0 <= (g.bbox[1] + g.bbox[3]) / 2 <= y1
               for g in line.glyphs) / len(line.glyphs)


def associate(lines: list[TextLine], regions: list[Region]) -> None:
    excluded = set(LABELS) - {"text", "abstract", "content", "formula", "formula_number"}
    for line in lines:
        matches = [(coverage(line, r), r) for r in regions]
        matches = [(c, r) for c, r in matches if c >= .65]
        if not matches:
            continue
        # A high-coverage excluded region wins over a surrounding text box.
        _, region = max(matches, key=lambda pair: (pair[1].label in excluded, pair[0] * pair[1].score))
        line.region_id, line.region_score, line.region_label = region.id, region.score, region.label
        line.kind = "excluded" if region.label in excluded else "math" if region.label.startswith("formula") else "body"
        line.reason = f"model:{region.label}"


def _join_glyphs(target: TextLine, fragment: TextLine, repair: str) -> None:
    target.glyphs.extend(fragment.glyphs)
    target.text = glyph_text(target.glyphs)
    target.bbox = enclosing([glyph.bbox for glyph in target.glyphs])
    target.repairs.append(f"{repair}:{fragment.id}")


def merge_baselines(lines: list[TextLine], body_size: float) -> list[TextLine]:
    merged: list[TextLine] = []
    for fragment in sorted(lines, key=lambda line: (line.baseline, line.bbox[0])):
        near = [line for line in merged[-12:]
                if abs(line.baseline - fragment.baseline) < max(1.8, body_size * .18)
                and (line.kind == fragment.kind or {line.kind, fragment.kind} <= {"body", "math", "unknown"})
                and (fragment.bbox[0] <= line.bbox[2] + body_size * 2 or fragment.text in {"□", "■", "◻", "◼", "▢", "∎"})]
        if near:
            target = min(near, key=lambda line: abs(line.baseline - fragment.baseline))
            _join_glyphs(target, fragment, "baseline_join")
            if fragment.kind == "body" and target.region_label not in {"text", "abstract", "content"}:
                target.region_id, target.region_score, target.region_label = fragment.region_id, fragment.region_score, fragment.region_label
                target.reason = fragment.reason
            target.kind = "body" if fragment.kind == "body" else target.kind
        else:
            merged.append(fragment)
    return merged


def attach_inline(merged: list[TextLine], body_size: float) -> None:
    for fragment in list(merged):
        if fragment.kind == "excluded" or len(fragment.text) > 25:
            continue
        # A large prose row must support the attachment; display fractions stay events.
        targets = [line for line in merged if line is not fragment and line.kind == "body"
                   and len(line.text) >= 25 and line.size >= body_size * .9
                   and abs(line.baseline - fragment.baseline) < body_size * .9
                   and fragment.bbox[2] >= line.bbox[0] - body_size * 3
                   and fragment.bbox[0] <= line.bbox[2] + body_size * 3]
        if targets:
            target = min(targets, key=lambda line: abs(line.baseline - fragment.baseline))
            _join_glyphs(target, fragment, "inline_fragment")
            merged.remove(fragment)
