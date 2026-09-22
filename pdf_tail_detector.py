#!/usr/bin/env python3
"""Find paragraphs whose last visible character reaches the right side of a PDF.

The detector is intentionally geometry based.  It uses pdfplumber character
boxes, groups them into text lines, then joins nearby lines into paragraphs
inside each column.  The default rule is ``last_char.x1 < page.width * 2/3``.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pdfplumber
from pypdf import PdfReader, PdfWriter
from reportlab.lib.colors import Color, red, yellow
from reportlab.pdfgen.canvas import Canvas


SECTION_RE = re.compile(r"^(?:\d+|[IVX]+)(?:\.\d+)*[.)]?\s+")
CAPTION_RE = re.compile(r"^(?:figure|fig\.?|table|algorithm|appendix)\s*\d*\s*[:.)-]?", re.I)
PAGE_NUMBER_RE = re.compile(r"^(?:\d+|[-–—]?\s*\d+\s*[-–—]?)$")


@dataclass
class Line:
    chars: list[dict[str, Any]]
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    column: str
    font_size: float


@dataclass
class Finding:
    page: int
    paragraph: int
    column: str
    paragraph_text: str
    final_line: str
    last_char: str
    last_char_x0: float
    last_char_x1: float
    page_width: float
    page_height: float
    threshold_x: float
    ratio: float
    template: str
    context_before: str = ""
    context_after: str = ""


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _median(values: Iterable[float], default: float) -> float:
    vals = [v for v in values if math.isfinite(v) and v > 0]
    return statistics.median(vals) if vals else default


def _column_for(x0: float, x1: float, width: float) -> str:
    """Classify by the left edge so short final lines stay in their column."""
    # A full-width line and a short line in a one-column paper have different
    # x1 values but the same x0.  Classifying by x1 would split one paragraph.
    return "right" if x0 >= width * 0.48 else "left"


def _chars_to_text(chars: list[dict[str, Any]]) -> str:
    """Recover ordinary word spaces from the gaps in PDFs with no space glyphs."""
    result: list[str] = []
    previous = None
    for char in chars:
        text = str(char.get("text", ""))
        if not text.strip():
            continue
        if previous is not None:
            gap = _num(char.get("x0")) - _num(previous.get("x1"))
            prev_text = str(previous.get("text", ""))
            if gap > 1.5 and prev_text not in "([{\"'" and text not in ",.;:!?)]}\"'":
                result.append(" ")
        result.append(text)
        previous = char
    return "".join(result).strip()


def _make_lines(page: pdfplumber.page.Page) -> list[Line]:
    chars = [c for c in page.chars if str(c.get("text", "")).strip()]
    if not chars:
        return []
    # A line groups glyphs with nearly the same top coordinate.  The tolerance
    # is relative to the page but bounded so small superscripts do not split a
    # normal text line into many fragments.
    tolerance = max(1.2, min(3.0, page.height * 0.0025))
    chars.sort(key=lambda c: (_num(c.get("top")), _num(c.get("x0"))))
    buckets: list[list[dict[str, Any]]] = []
    means: list[float] = []
    for char in chars:
        top = _num(char.get("top"))
        best = None
        for i, mean in enumerate(means):
            if abs(top - mean) <= tolerance:
                best = i
                break
        if best is None:
            buckets.append([char])
            means.append(top)
        else:
            buckets[best].append(char)
            means[best] = sum(_num(c.get("top")) for c in buckets[best]) / len(buckets[best])

    lines: list[Line] = []
    for bucket in buckets:
        bucket.sort(key=lambda c: _num(c.get("x0")))
        # ICLR/NeurIPS review PDFs may contain line numbers in the left
        # margin. Remove a short numeric prefix when it is separated from
        # the body by a visible horizontal gap.
        if len(bucket) >= 2:
            prefix_end = 0
            while prefix_end < len(bucket) and str(bucket[prefix_end].get("text", "")).strip().isdigit():
                prefix_end += 1
            if 0 < prefix_end < len(bucket) and prefix_end <= 4:
                gap = _num(bucket[prefix_end].get("x0")) - _num(bucket[prefix_end - 1].get("x1"))
                if _num(bucket[0].get("x0")) < page.width * 0.20 and gap >= 5.0:
                    bucket = bucket[prefix_end:]
        text = _chars_to_text(bucket)
        if not text:
            continue
        x0 = min(_num(c.get("x0")) for c in bucket)
        x1 = max(_num(c.get("x1")) for c in bucket)
        top = min(_num(c.get("top")) for c in bucket)
        bottom = max(_num(c.get("bottom"), _num(c.get("top"))) for c in bucket)
        sizes = [_num(c.get("size"), 0.0) for c in bucket]
        lines.append(Line(bucket, text, x0, x1, top, bottom,
                          _column_for(x0, x1, page.width), _median(sizes, 10.0)))
    return sorted(lines, key=lambda l: (l.top, l.x0))


def _looks_structural(line: Line, page: pdfplumber.page.Page, body_size: float) -> bool:
    text = line.text.strip()
    low = text.lower()
    if not text or PAGE_NUMBER_RE.fullmatch(text):
        return True
    # Running headers, page numbers, and footers are not body paragraphs.
    if line.top < page.height * 0.065 or line.bottom > page.height * 0.95:
        return True
    # Titles, section headings, captions, and labels are intentionally not
    # treated as paragraphs even if their final word happens to be long.
    if SECTION_RE.match(text) and len(text) <= 120:
        return True
    letters = sum(ch.isalpha() for ch in text)
    upper = sum(ch.isupper() for ch in text if ch.isalpha())
    if len(text) <= 120 and letters >= 4 and upper / max(1, letters) > 0.78:
        return True
    if CAPTION_RE.match(text):
        return True
    # Algorithm blocks are commonly monospaced or begin with pseudocode
    # keywords. They are not prose paragraphs.
    fonts = " ".join(str(c.get("fontname", "")) for c in line.chars).lower()
    monospace = any(token in fonts for token in ("courier", "typewriter", "mono", "cmtt", "pcr"))
    code_start = re.match(
        r"^(?:for\s+\w+\s+in\b|while\s*\(|if\s*\(|else\b|elif\b|return\b|repeat\b|until\b|end\b|"
        r"require\s*:|ensure\s*:|input\s*:|output\s*:|function\b|procedure\b)", text, re.I
    )
    if monospace or code_start or any(symbol in text for symbol in ("←", "<-", ":=")):
        return True
    # Display equations and isolated math lines generally contain very few
    # alphabetic characters.  Inline math remains part of ordinary prose.
    if len(text) >= 3 and letters < max(3, len(text) * 0.28) and any(ch in text for ch in "=∑∫√{}^_"):
        return True
    # Very small text is usually a footnote or template metadata.  Keep it
    # when it is close to body size; this avoids relying on a fixed point size.
    if body_size and line.font_size < body_size * 0.92:
        return True
    if low in {"abstract", "introduction", "references", "acknowledgments", "acknowledgements"}:
        return True
    return False


def _looks_table_like(text: str) -> bool:
    """Reject table rows and symbol lists that have no prose paragraph shape."""
    compact = text.replace(" ", "")
    if not compact:
        return True
    letters = sum(ch.isalpha() for ch in compact)
    digits = sum(ch.isdigit() for ch in compact)
    if digits / len(compact) > 0.28 and letters / len(compact) < 0.55:
        return True
    if compact.count("=") >= 2 and letters / len(compact) < 0.62:
        return True
    if compact.count("|") >= 1:
        return True
    return False


def _looks_math_like(text: str) -> bool:
    compact = text.replace(" ", "")
    if not compact:
        return False
    letters = sum(ch.isalpha() for ch in compact)
    return (
        compact.startswith("(cid")
        or letters / len(compact) < 0.58 and any(ch in compact for ch in "=≤≥∑∫√{}^_()")
    )


def _paragraphs(page: pdfplumber.page.Page, lines: list[Line], body_size: float | None = None) -> list[list[Line]]:
    if not lines:
        return []
    body_size = body_size or _median([l.font_size for l in lines if len(l.text) > 20], 10.0)
    by_column: dict[str, list[Line]] = {"left": [], "right": [], "full": []}
    for line in lines:
        by_column.setdefault(line.column, []).append(line)
    groups: list[list[Line]] = []
    for column_lines in by_column.values():
        column_lines.sort(key=lambda l: (l.top, l.x0))
        current: list[Line] = []
        skip_caption_block = False
        caption_bottom = 0.0
        for line in column_lines:
            if _looks_structural(line, page, body_size):
                if CAPTION_RE.match(line.text.strip()):
                    skip_caption_block = True
                    caption_bottom = line.bottom
                if current:
                    groups.append(current)
                    current = []
                continue
            if skip_caption_block:
                # Skip immediately following table/figure rows, but let a
                # normal prose paragraph resume after a visible gap.
                if line.top - caption_bottom > max(14.0, body_size * 1.8):
                    skip_caption_block = False
                elif _looks_table_like(line.text):
                    caption_bottom = line.bottom
                    continue
                elif line.text[:1].isupper() and len(line.text) > 45:
                    skip_caption_block = False
                else:
                    caption_bottom = line.bottom
                    continue
            if not current:
                current = [line]
                continue
            prev = current[-1]
            gap = line.top - prev.bottom
            typical_height = _median([x.bottom - x.top for x in current[-3:]], 10.0)
            same_indent = abs(line.x0 - prev.x0) <= max(18.0, typical_height * 1.8)
            # TeX paragraph spacing is normally below two line heights.  A
            # larger gap, a changed indent, or a column change starts a new
            # paragraph.
            if gap <= max(5.0, typical_height * 1.65) and same_indent:
                current.append(line)
            else:
                groups.append(current)
                current = [line]
        if current:
            groups.append(current)
    return sorted(groups, key=lambda p: (p[0].top, p[0].x0))


def scan_pdf(input_pdf: Path, ratio: float = 2 / 3, template: str = "auto") -> tuple[list[Finding], list[dict[str, Any]]]:
    findings: list[Finding] = []
    annotations: list[dict[str, Any]] = []
    references_started = False
    with pdfplumber.open(str(input_pdf)) as pdf:
        page_lines_all = [_make_lines(page) for page in pdf.pages]
        global_body_size = _median(
            [line.font_size for lines in page_lines_all for line in lines if len(line.text) > 35],
            10.0,
        )
        for page_no, (page, page_lines) in enumerate(zip(pdf.pages, page_lines_all), start=1):
            threshold = page.width * ratio
            page_lines = _make_lines(page)
            if any(re.match(r"^references\b", line.text.strip(), re.I) for line in page_lines):
                references_started = True
            if references_started:
                continue
            paragraphs = _paragraphs(page, page_lines, global_body_size)
            for para_no, para in enumerate(paragraphs, start=1):
                # A paragraph should have enough prose to distinguish it from
                # a heading or a one-line label.  Single-line body paragraphs
                # are retained when they are long enough.
                paragraph_text = " ".join(line.text for line in para).strip()
                if len(paragraph_text) < 35:
                    continue
                if _looks_table_like(paragraph_text) or CAPTION_RE.match(paragraph_text):
                    continue
                final = para[-1]
                visible_chars = [c for c in final.chars if str(c.get("text", "")).strip()]
                if not visible_chars:
                    continue
                last = visible_chars[-1]
                x0, x1 = _num(last.get("x0")), _num(last.get("x1"))
                if x1 >= threshold:
                    continue
                following = [
                    candidate.text for candidate in page_lines
                    if candidate.column == final.column
                    and candidate.top > final.bottom + 0.5
                    and not PAGE_NUMBER_RE.fullmatch(candidate.text.strip())
                ]
                next_text = following[0] if following else ""
                # A prose line ending in ':' or 'is' immediately before a
                # display equation is a lead-in, not a short paragraph tail.
                if next_text and _looks_math_like(next_text) and (
                    final.text.rstrip().endswith((":", ";"))
                    or re.search(r"\b(?:is|are|be|then)$", final.text.rstrip(), re.I)
                ):
                    continue
                finding = Finding(
                    page=page_no,
                    paragraph=para_no,
                    column=final.column,
                    paragraph_text=paragraph_text,
                    final_line=final.text,
                    last_char=str(last.get("text", "")),
                    last_char_x0=round(x0, 3),
                    last_char_x1=round(x1, 3),
                    page_width=round(page.width, 3),
                    page_height=round(page.height, 3),
                    threshold_x=round(threshold, 3),
                    ratio=ratio,
                    template=template,
                    context_before=para[-2].text if len(para) > 1 else "",
                    context_after=following[0] if following else "",
                )
                findings.append(finding)
                annotations.append({
                    "page": page_no,
                    "x0": max(0.0, min(page.width, final.x0 - 1.5)),
                    "x1": max(0.0, min(page.width, final.x1 + 1.5)),
                    "top": max(0.0, final.top - 1.5),
                    "bottom": min(page.height, final.bottom + 1.5),
                    "threshold": threshold,
                    "last_char_x1": x1,
                })
    return findings, annotations


def _write_annotated(input_pdf: Path, output_pdf: Path, annotations: list[dict[str, Any]], ratio: float) -> None:
    reader = PdfReader(str(input_pdf))
    writer = PdfWriter()
    by_page: dict[int, list[dict[str, Any]]] = {}
    for ann in annotations:
        by_page.setdefault(int(ann["page"]), []).append(ann)
    for idx, page in enumerate(reader.pages, start=1):
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        overlay_path = output_pdf.with_name(f".{output_pdf.stem}-overlay-{idx}.pdf")
        canvas = Canvas(str(overlay_path), pagesize=(width, height))
        canvas.setStrokeColor(Color(1, 0.55, 0, alpha=0.8))
        canvas.setDash(4, 3)
        threshold = width * ratio
        canvas.line(threshold, 0, threshold, height)
        canvas.setDash()
        for ann in by_page.get(idx, []):
            y = height - ann["bottom"]
            h = ann["bottom"] - ann["top"]
            canvas.setStrokeColor(red)
            canvas.setFillColor(Color(1, 0, 0, alpha=0.08))
            canvas.rect(ann["x0"], y, ann["x1"] - ann["x0"], h, fill=1, stroke=1)
            canvas.setFillColor(yellow)
            canvas.setFont("Helvetica", 6)
            canvas.drawString(ann["x0"], y + h + 1, f"tail x={ann['last_char_x1']:.1f}")
        canvas.save()
        overlay = PdfReader(str(overlay_path)).pages[0]
        page.merge_page(overlay)
        writer.add_page(page)
        overlay_path.unlink(missing_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with output_pdf.open("wb") as stream:
        writer.write(stream)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect short-tail paragraphs in ICLR/NeurIPS-style PDFs.")
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="annotated PDF path (default: <input>.tail-marked.pdf)")
    parser.add_argument("--ratio", type=float, default=2 / 3, help="right-edge threshold as a fraction of page width (default: 0.666667)")
    parser.add_argument("--template", choices=["auto", "iclr", "neurips", "nips"], default="auto")
    args = parser.parse_args()
    if not 0 < args.ratio < 1:
        parser.error("--ratio must be between 0 and 1")
    if not args.input_pdf.exists():
        parser.error(f"input does not exist: {args.input_pdf}")
    output_pdf = args.output or args.input_pdf.with_name(f"{args.input_pdf.stem}.tail-marked.pdf")
    findings, annotations = scan_pdf(args.input_pdf, args.ratio, args.template)
    _write_annotated(args.input_pdf, output_pdf, annotations, args.ratio)
    print(f"短尾段数量: {len(findings)}")
    print(f"标记规则: 最后可见字符右边界 x1 < 页面宽度 × {args.ratio:g}")
    print(f"标记 PDF: {output_pdf.resolve()}")
    for index, finding in enumerate(findings, start=1):
        print(f"\n[{index}] 第 {finding.page} 页 / {finding.column} 栏 / 段落 {finding.paragraph}")
        print(f"    末字: {finding.last_char!r}; x1={finding.last_char_x1:.2f}; 阈值={finding.threshold_x:.2f}")
        if finding.context_before:
            print(f"    上一行: {finding.context_before}")
        print(f"    末行:   {finding.final_line}")
        if finding.context_after:
            print(f"    后一行: {finding.context_after}")
        print(f"    段落:   {finding.paragraph_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
