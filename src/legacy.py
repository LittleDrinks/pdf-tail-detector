#!/usr/bin/env python3
"""Find short paragraph tails in PDFs using PyMuPDF text blocks and lines."""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pymupdf


SECTION_RE = re.compile(r"^(?:\d+|[IVX]+)(?:\.\d+)*[.)]?\s+")
CAPTION_RE = re.compile(r"^(?:figure|fig\.?|table|algorithm|appendix)\s*\d*\s*[:.)-]?", re.I)
PAGE_NUMBER_RE = re.compile(r"^(?:\d+|[-–—]?\s*\d+\s*[-–—]?)$")
ALGORITHM_HEADING_RE = re.compile(r"^algorithm\s+\d+\b", re.I)
ALGORITHM_END_RE = re.compile(r"^(?:\d+\s*:\s*)?end\s+for\b", re.I)
ALGORITHM_STEP_RE = re.compile(r"^\d+\s*:\s*")
ANNOTATION_RE = re.compile(r"^tail\s+x\s*=", re.I)
MATH_FONT_RE = re.compile(r"(?:cmmi|cmsy|cmex|msam|msbm|cmr\d|math|symbol)", re.I)
FORMULA_LEAD_RE = re.compile(
    r"(?:\b(?:define|satisf(?:y|ies)|give|gives|yield|yields|form|follows|"
    r"transformation|assume|consider|suppose|let|write|set|has|have|is|are|be|then)"
    r"|law|:)\s*$|[,;]\s*$",
    re.I,
)


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
    last_char_x1: float
    threshold_x: float
    bbox: tuple[float, float, float, float]
    context_before: str = ""
    context_after: str = ""


def _median(values: Iterable[float], default: float) -> float:
    vals = [v for v in values if math.isfinite(v) and v > 0]
    return statistics.median(vals) if vals else default


def _column_for(x0: float, width: float) -> str:
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
            gap = char["x0"] - previous["x1"]
            prev_text = str(previous.get("text", ""))
            if gap > 1.5 and prev_text not in "([{\"'" and text not in ",.;:!?)]}\"'":
                result.append(" ")
        result.append(text)
        previous = char
    return "".join(result).strip()


def _make_blocks(page: pymupdf.Page) -> list[list[Line]]:
    blocks: list[list[Line]] = []
    for raw_block in page.get_text("rawdict", sort=True)["blocks"]:
        if raw_block["type"] != 0:
            continue
        lines: list[Line] = []
        for raw_line in raw_block["lines"]:
            chars = []
            for span in raw_line["spans"]:
                for char in span.get("chars", []):
                    x0, top, x1, bottom = char["bbox"]
                    chars.append({"text": char["c"], "x0": x0, "x1": x1,
                                  "top": top, "bottom": bottom,
                                  "size": span["size"], "fontname": span["font"]})
            visible = [char for char in chars if char["text"].strip()]
            if not visible:
                continue
            visible.sort(key=lambda char: char["x0"])
            x0 = min(char["x0"] for char in visible)
            x1 = max(char["x1"] for char in visible)
            top = min(char["top"] for char in visible)
            bottom = max(char["bottom"] for char in visible)
            line = Line(visible, _chars_to_text(visible), x0, x1, top, bottom,
                        _column_for(x0, page.rect.width),
                        _median([char["size"] for char in visible], 10.0))
            if lines and abs(line.top - lines[-1].top) < 2.0:
                previous = lines[-1]
                previous.chars.extend(line.chars)
                previous.chars.sort(key=lambda char: char["x0"])
                previous.text = _chars_to_text(previous.chars)
                previous.x0 = min(previous.x0, line.x0)
                previous.x1 = max(previous.x1, line.x1)
                previous.top = min(previous.top, line.top)
                previous.bottom = max(previous.bottom, line.bottom)
                previous.font_size = _median([char["size"] for char in previous.chars], 10.0)
            else:
                lines.append(line)
        if lines:
            blocks.append(lines)
    return blocks


def _looks_structural(line: Line, page: pymupdf.Rect, body_size: float) -> bool:
    text = line.text.strip()
    low = text.lower()
    if not text or PAGE_NUMBER_RE.fullmatch(text) or ANNOTATION_RE.match(text):
        return True
    if "@" in text or re.match(r"^\([a-z]\)\s", text):
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
    # Very small text is usually a footnote or template metadata.  Keep it
    # when it is close to body size; this avoids relying on a fixed point size.
    if body_size and line.font_size < body_size * 0.92:
        # TeX emits inline superscripts and subscripts as separate, smaller
        # rows. They belong to the surrounding paragraph; page numbers and
        # footnotes use non-math fonts and remain structural.
        if not any(MATH_FONT_RE.search(str(c.get("fontname", ""))) for c in line.chars):
            return True
    if low in {"abstract", "introduction", "references", "acknowledgments", "acknowledgements"}:
        return True
    return False


def _is_display_math_line(line: Line, page: pymupdf.Rect, body_size: float) -> bool:
    """Recognize TeX display-math fragments without rejecting inline math."""
    text = line.text.strip()
    if len(text) < 1:
        return False
    visible = [c for c in line.chars if str(c.get("text", "")).strip()]
    if not visible:
        return False
    math_chars = sum(1 for c in visible if MATH_FONT_RE.search(str(c.get("fontname", ""))))
    math_ratio = math_chars / len(visible)
    compact = text.replace(" ", "")
    letters = sum(ch.isalpha() for ch in compact)
    symbol_heavy = any(ch in compact for ch in "=≤≥∑∫√{}^_∥()")
    # Standalone equation rows are usually dominated by CM
    # math fonts, or carry a cid glyph marker from a symbol font. Prose rows
    # with inline variables have a much lower math-font ratio.
    if math_ratio >= 0.65 and (symbol_heavy or len(text) <= 32):
        return True
    if "(cid:" in text and math_ratio >= 0.45 and len(text) < 90:
        return True
    if math_ratio >= 0.80 and len(text) < 70:
        return True
    if letters < max(3, len(compact) * 0.45) and math_ratio >= 0.35 and symbol_heavy:
        return True
    # A narrow, small-font equation row is often made entirely of symbols
    # even when the embedded font name is not preserved by the PDF producer.
    if body_size and line.font_size < body_size * 0.9 and line.x1 - line.x0 < page.width * 0.62:
        if letters < max(4, len(compact) * 0.55) and symbol_heavy:
            return True
    return False


def _without_algorithm_blocks(lines: list[Line]) -> list[Line]:
    """Drop complete algorithm environments, including their step rows."""
    result: list[Line] = []
    in_algorithm = False
    after_end = False
    last_algorithm_top = 0.0
    for index, line in enumerate(lines):
        text = line.text.strip()
        if ALGORITHM_HEADING_RE.match(text):
            upcoming = (candidate.text.strip() for candidate in lines[index + 1:index + 12])
            if any(ALGORITHM_STEP_RE.match(row) or re.match(r"^(?:require|ensure|input|output)\s*:", row, re.I)
                   for row in upcoming):
                in_algorithm = True
                after_end = False
                continue
        if in_algorithm:
            if after_end and not PAGE_NUMBER_RE.fullmatch(text):
                # Algorithms may contain several independent loops. The
                # first ``end for`` is not the end of the environment; the
                # final one is followed by a visible prose-sized gap.
                if line.top - last_algorithm_top > 20.0:
                    in_algorithm = False
                    after_end = False
                    result.append(line)
                    continue
            if ALGORITHM_END_RE.match(text):
                after_end = True
                last_algorithm_top = line.top
            elif not PAGE_NUMBER_RE.fullmatch(text):
                last_algorithm_top = line.top
            continue
        result.append(line)
    return result


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


def _has_rightward_continuation(line: Line, page_lines: list[Line], body_size: float) -> bool:
    center = (line.top + line.bottom) / 2
    return any(
        other is not line
        and other.x0 >= line.x1 - 5.0
        and other.x1 > line.x1 + 2.0
        and abs((other.top + other.bottom) / 2 - center) < max(3.0, body_size * 0.5)
        for other in page_lines
    )


def scan_pdf(input_pdf: Path, ratio: float = 2 / 3) -> list[Finding]:
    findings: list[Finding] = []
    references_started = False
    with pymupdf.open(input_pdf) as pdf:
        page_blocks_all = [_make_blocks(page) for page in pdf]
        global_body_size = _median(
            [line.font_size for blocks in page_blocks_all for block in blocks
             for line in block if len(line.text) > 35],
            10.0,
        )
        for page_no, (page, blocks) in enumerate(zip(pdf, page_blocks_all), start=1):
            if references_started:
                break
            page_size = page.rect
            threshold = page_size.width * ratio
            page_lines = sorted((line for block in blocks for line in block), key=lambda line: (line.top, line.x0))
            if page_no == 1:
                abstract = next((line for line in page_lines if re.sub(r"\W", "", line.text).lower() == "abstract"), None)
                if abstract:
                    page_lines = [line for line in page_lines if line.top >= abstract.top]
            for index, line in enumerate(page_lines):
                title = line.text
                if title.strip().upper() == "R" and index + 1 < len(page_lines):
                    next_line = page_lines[index + 1]
                    if next_line.top - line.top < 5:
                        title += next_line.text
                if re.sub(r"\W", "", title).upper() == "REFERENCES":
                    page_lines = page_lines[:index]
                    references_started = True
                    break
            # Keep the raw rows for detecting a prose lead-in immediately
            # followed by display math, but never offer algorithm rows to the
            # paragraph builder.
            raw_page_lines = page_lines
            page_lines = _without_algorithm_blocks(page_lines)
            retained = {id(line) for line in page_lines}
            paragraphs: list[list[Line]] = []
            for block in blocks:
                if block and CAPTION_RE.match(block[0].text):
                    continue
                current: list[Line] = []
                for line in block:
                    if id(line) not in retained or _looks_structural(line, page_size, global_body_size):
                        if current:
                            paragraphs.append(current)
                            current = []
                    else:
                        current.append(line)
                if current:
                    paragraphs.append(current)
            paragraphs.sort(key=lambda para: (para[0].top, para[0].x0))
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
                if _is_display_math_line(final, page_size, global_body_size):
                    continue
                if _has_rightward_continuation(final, raw_page_lines, global_body_size):
                    continue
                last = visible_chars[-1]
                x1 = last["x1"]
                if x1 >= threshold:
                    continue
                following_rows = [
                    candidate for candidate in raw_page_lines
                    if candidate.column == final.column
                    and candidate.top > final.bottom + 0.5
                    and not PAGE_NUMBER_RE.fullmatch(candidate.text.strip())
                    and not ANNOTATION_RE.match(candidate.text.strip())
                ]
                following = [candidate.text for candidate in following_rows]
                # A prose line ending in ':' or 'is' immediately before a
                # display equation is a lead-in, not a short paragraph tail.
                next_row = following_rows[0] if following_rows else None
                if next_row and _is_display_math_line(next_row, page_size, global_body_size) and FORMULA_LEAD_RE.search(final.text):
                    continue
                finding = Finding(
                    page=page_no,
                    paragraph=para_no,
                    column=final.column,
                    paragraph_text=paragraph_text,
                    final_line=final.text,
                    last_char=str(last.get("text", "")),
                    last_char_x1=round(x1, 3),
                    threshold_x=round(threshold, 3),
                    bbox=(
                        max(0.0, final.x0 - 1.5),
                        min(page_size.width, final.x1 + 1.5),
                        max(0.0, final.top - 1.5),
                        min(page_size.height, final.bottom + 1.5),
                    ),
                    context_before=para[-2].text if len(para) > 1 else "",
                    context_after=following[0] if following else "",
                )
                findings.append(finding)
    return findings
