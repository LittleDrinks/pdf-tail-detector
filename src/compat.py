"""Keep the scan_pdf return shape while using the shared geometry pipeline."""
from dataclasses import dataclass
from pathlib import Path

from .pipeline import analyze_pdf


@dataclass
class Finding:
    page: int
    paragraph: int
    column: str
    paragraph_text: str
    final_line: str
    last_char: str
    last_char_x1: float
    threshold_x: float
    bbox: tuple[float, float, float, float]
    context_before: str = ""
    context_after: str = ""


def _finding(item: dict) -> Finding:
    x0, y0, x1, y1 = item["bbox"]
    return Finding(item["page"], int(item["paragraph_id"].removeprefix("para-")), "left",
                   item["paragraph_text"], item["final_line"], item["last_char"],
                   item["last_char_x1"], item["threshold_x"], (x0, x1, y0, y1), item["context_before"])


def scan_pdf(input_pdf: Path, ratio: float = 2 / 3) -> list[Finding]:
    report = analyze_pdf(input_pdf, backend="none", page_ratio=ratio)
    return [_finding(item) for item in report["candidates"]]
