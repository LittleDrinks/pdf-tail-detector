"""Recover paragraph boundaries from a stream of prose and structural events."""
from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field

from .layout import PageData, Rect, TextLine, enclosing, glyph_text, merge_visual_lines


MATH_FONT = re.compile(r"cmmi|cmsy|cmex|msam|msbm|math|symbol", re.I)
ENTITY = re.compile(r"^(?:Theorem|Lemma|Corollary|Proposition|Definition|Condition|Proof)\b", re.I)
QED = {"□", "■", "◻", "◼", "▢", "∎"}


@dataclass
class Paragraph:
    id: str
    lines: list[TextLine]
    end_reason: str
    status: str = "checked"
    math_ending: bool = False
    evidence: list[str] = field(default_factory=list)


def body_size(pages: list[PageData]) -> float:
    sizes = [g.size for page in pages for line in page.lines if len(line.text) > 35
             for g in line.glyphs if g.text.isalpha() and not MATH_FONT.search(g.font)]
    return statistics.median(sizes) if sizes else 10.


def normalized(text: str) -> str:
    return re.sub(r"[^A-Za-z]", "", text).lower()


def refine_page(page: PageData, size: float, model: bool) -> None:
    # These are supplemental cues. Region exclusions remain visible in the report.
    for line in page.lines:
        text = line.text.strip()
        low = normalized(text)
        words = re.findall(r"[A-Za-z]{3,}", text)
        math_ratio = sum(bool(MATH_FONT.search(g.font)) for g in line.glyphs) / len(line.glyphs)
        if line.kind == "body" and math_ratio > .6 and len(words) < 2 and len(text) < 25:
            line.kind, line.reason = "math", "inline_math_fragment"
        if line.kind == "math" and len(words) >= 4 and math_ratio < .45:
            line.kind, line.reason = "body", "prose_overlaps_formula"
            line.repairs.append("formula_region_conflict")
        if line.kind == "excluded":
            continue
        margin_digit = bool(re.fullmatch(r"\d+", text) and
                            (line.bbox[0] < page.width * .15 or line.bbox[1] > page.height * .92))
        if margin_digit or line.bbox[1] < page.height * .065 or line.bbox[3] > page.height * .95:
            line.kind, line.reason = "excluded", "margin_or_page_number"
        elif text.lower().startswith("tail x="):
            line.kind, line.reason = "excluded", "legacy_annotation"
        elif low in {"abstract", "references", "acknowledgments", "acknowledgements", "introduction"}:
            line.kind, line.reason = "excluded", "section_heading"
        elif re.match(r"^(?:\d+(?:\.\d+)*\s+|[A-Z]\s+)", text) and len(text) < 120 and sum(c.isupper() for c in text) > sum(c.islower() for c in text):
            line.kind, line.reason = "excluded", "section_heading"
        elif line.size < size * .92 and math_ratio < .2 and len(words) >= 3:
            line.kind, line.reason = "excluded", "small_prose_footnote"
        elif re.match(r"^(?:Algorithm\s+\d|\d+\s*:|Require:|Ensure:|Input:|Output:)", text, re.I):
            line.kind, line.reason = "excluded", "algorithm_cue"
        elif re.match(r"^[•●◦]\s*|^\([a-z]\)\s+", text):
            line.kind, line.reason = "excluded", "list_item"
        elif line.kind == "unknown":
            if math_ratio > .6 or (len(words) < 2 and any(c in text for c in "=∑∫≤≥")):
                line.kind, line.reason = "math", "math_geometry"
            elif (len(words) >= 3 or (len(words) >= 2 and len(text) >= 12)) and line.size >= size * .9:
                line.kind = "body"
                line.reason = "heuristic_uncovered_prose" if model else "geometry_baseline_prose"
    for line in page.lines:
        if line.kind != "unknown" or line.size < size * .9:
            continue
        neighbors = [other for other in page.lines if other.kind == "body"
                     and abs(other.bbox[0] - line.bbox[0]) < 3
                     and 0 < line.baseline - other.baseline <= size * 1.5]
        if neighbors and any(c.isalpha() for c in line.text):
            line.kind, line.reason = "body", "aligned_short_prose"
    # Supplemental algorithm/list ranges end at a visible gap, not at 'end for'.
    ordered = sorted(page.lines, key=lambda line: (line.baseline, line.bbox[0]))
    previous = None
    list_active = False
    algorithm_active = False
    for line in ordered:
        gap = line.baseline - previous.baseline if previous else 0
        if gap > size * 1.65 and line.reason not in {"list_item", "algorithm_cue"}:
            list_active = algorithm_active = False
        if line.reason == "list_item":
            list_active = True
        if re.match(r"^Algorithm\s+\d", line.text, re.I):
            algorithm_active = True
        if list_active and line.kind != "excluded":
            line.kind, line.reason = "excluded", "list_continuation"
        if algorithm_active and line.kind != "excluded":
            line.kind, line.reason = "excluded", "algorithm_continuation"
        previous = line
    page.lines = merge_visual_lines(page.lines, size)
    # Title and author metadata precede Abstract; do not depend on font names.
    if page.number == 1:
        abstract = next((line for line in page.lines if normalized(line.text) == "abstract"), None)
        if abstract:
            for line in page.lines:
                if line.baseline < abstract.baseline:
                    line.kind, line.reason = "excluded", "front_matter"


def leading(lines: list[TextLine], size: float) -> float:
    values = [b.baseline - a.baseline for a, b in zip(lines, lines[1:])
              if a.kind == b.kind == "body" and size * .9 < b.baseline - a.baseline < size * 1.4]
    return statistics.median(values) if values else size * 1.1


def aligned_value(values: list[float], default: float) -> tuple[float, int]:
    if not values:
        return default, 0
    cluster, count = Counter(round(v / 2) for v in values).most_common(1)[0]
    members = [v for v in values if abs(v - cluster * 2) <= 2]
    return statistics.median(members), count


def document_bounds(pages: list[PageData]) -> tuple[float, float, int]:
    lines = [line for page in pages for line in page.lines if line.kind == "body"
             and line.bbox[2] - line.bbox[0] > page.width * .48 and line.region_label != "abstract"]
    left, support = aligned_value([line.bbox[0] for line in lines], 0)
    right, nright = aligned_value([line.bbox[2] for line in lines], 0)
    return left, right, min(support, nright)


def multi_column(page: PageData) -> bool:
    # Require simultaneous prose rows in both columns, not just a narrow abstract.
    prose = [line for line in page.lines if line.kind == "body" and len(line.text) > 30]
    pairs = sum(1 for a in prose for b in prose if a.bbox[0] < page.width * .35
                and b.bbox[0] > page.width * .48 and a.bbox[2] < b.bbox[0] - 8
                and abs(a.baseline - b.baseline) < 3)
    return pairs >= 3


def paragraph_bounds(para: Paragraph, page: PageData, global_bounds: tuple[float, float, int]) -> dict:
    prose = [line for line in para.lines if line.kind == "body" and line.page == page.number]
    left, right, support = global_bounds
    source = "document_alignment"
    region = next((r for r in page.regions if prose and r.id == prose[-1].region_id), None)
    abstract = bool(prose and prose[-1].region_label == "abstract")
    around_image = bool(region and any(r.label in {"image", "chart", "table"}
                                      and r.bbox[1] < region.bbox[3] and r.bbox[3] > region.bbox[1]
                                      and r.bbox[0] >= region.bbox[2] - 10 for r in page.regions))
    if abstract or around_image:
        local = [line for line in page.lines if line.kind == "body"
                 and (line.region_id == region.id if region else line in prose)]
        if len(local) >= 2:
            left = min(line.bbox[0] for line in local)
            right = max(line.bbox[2] for line in local)
            support, source = len(local), "abstract_alignment" if abstract else "wrap_alignment"
    if right <= left or support < 2:
        long = [line for line in prose if len(line.text) >= 45]
        if len(long) >= 2:
            left = min(line.bbox[0] for line in long)
            right = max(line.bbox[2] for line in long)
            support, source = len(long), "paragraph_alignment"
    return {"left": left, "right": right, "support": support, "source": source,
            "reliable": right > left and support >= 2}


def recover_paragraphs(pages: list[PageData], size: float) -> list[Paragraph]:
    paragraphs: list[Paragraph] = []
    current: list[TextLine] = []
    last_event: TextLine | None = None
    math_ending = False
    previous_page: PageData | None = None
    evidence: list[str] = []

    def finish(reason: str, status: str = "checked") -> None:
        nonlocal current, math_ending, evidence
        if current:
            paragraphs.append(Paragraph(f"para-{len(paragraphs) + 1}", current, reason, status, math_ending, evidence))
        current, evidence, math_ending = [], [], False

    for page in pages:
        if page.status != "checked":
            finish("unsupported_page_boundary", "review")
            previous_page = page
            last_event = None
            continue
        step = leading(page.lines, size)
        first_body = next((line for line in page.lines if line.kind == "body"), None)
        if current and previous_page:
            last = current[-1]
            before_body = page.lines[:page.lines.index(first_body)] if first_body else page.lines
            blocking = any(line.reason == "section_heading" or line.region_label == "paragraph_title" for line in before_body)
            compatible = bool(first_body and abs(first_body.bbox[0] - last.bbox[0]) < size * 1.5
                              and abs(first_body.size - last.size) < size * .15)
            lower = bool(first_body and re.match(r"^[a-z(]", first_body.text))
            unfinished = not re.search(r"[.!?]\s*[)\]”\"']?$", last.text)
            if page.number == previous_page.number + 1 and compatible and not blocking and (unfinished or lower):
                evidence.append(f"cross_page:{previous_page.number}->{page.number}")
            elif blocking or not unfinished:
                finish("page_boundary_complete")
            else:
                finish("ambiguous_page_continuation", "review")
            last_event = None
        for line in page.lines:
            if line.kind == "excluded":
                if line.reason in {"margin_or_page_number", "small_prose_footnote"} or line.region_label in {"header", "footer", "footnote", "number"}:
                    continue
                # A caption alongside a wrap-around prose column does not interrupt it.
                if current and line.bbox[0] > max(l.bbox[2] for l in current if l.page == current[-1].page) + 8:
                    continue
                if current:
                    complete = bool(re.search(r"[.!?]\s*$", current[-1].text))
                    finish("structural_boundary", "checked" if complete or math_ending else "review")
                last_event = line
                continue
            if line.kind == "unknown":
                if current:
                    evidence.append(f"unknown_fragment:{line.id}")
                last_event = line
                continue
            if line.kind == "math":
                if current:
                    math_ending = True
                    evidence.append(f"display_math:p{line.page}:{line.id}")
                last_event = line
                continue
            if current and last_event and last_event.page == line.page:
                gap = line.baseline - last_event.baseline
                bold_prefix = any(re.search(r"bold|medi|cmbx|cmssbx", g.font, re.I) for g in line.glyphs[:12])
                entity_start = bool(ENTITY.match(line.text) and (bold_prefix or re.match(r"^Proof[.:]", line.text)))
                indent = (line.bbox[0] - min(l.bbox[0] for l in current) > size * .85
                          and len(re.findall(r"[A-Za-z]{3,}", line.text)) >= 4
                          and bool(re.search(r"[.!?]\s*$", current[-1].text)))
                normal_gap = gap > max(step * 1.35, step + size * .35)
                after_formula_gap = last_event.kind == "math" and gap > step * 1.65
                if entity_start or (last_event.kind != "math" and (normal_gap or indent)) or after_formula_gap:
                    finish("entity_start" if entity_start else "paragraph_spacing" if normal_gap else "indent_or_formula_gap")
            current.append(line)
            math_ending = False
            last_event = line
        previous_page = page
    finish("document_end")
    return paragraphs


def measure(para: Paragraph, page: PageData, bounds: dict, ratio: float,
            page_ratio: float | None = None) -> dict:
    line = para.lines[-1]
    glyphs = list(line.glyphs)
    corrections = list(line.repairs)
    # A proof-ending square is peripheral only when separated from actual prose.
    ordered = sorted(glyphs, key=lambda g: g.bbox[0])
    if ordered and ordered[-1].text in QED and re.search(r"\bProof\b", " ".join(l.text for l in para.lines), re.I):
        marker = ordered[-1]
        others = [g for g in glyphs if g is not marker]
        if others and marker.bbox[0] - max(g.bbox[2] for g in others) > line.size * 1.8:
            glyphs = others
            corrections.append("removed_isolated_proof_qed")
    effective = enclosing([g.bbox for g in glyphs])
    rightmost = max(glyphs, key=lambda glyph: glyph.bbox[2])
    left, right = bounds["left"], bounds["right"]
    threshold = page.width * page_ratio if page_ratio is not None else left + ratio * (right - left)
    status = para.status
    if not bounds["reliable"] and page_ratio is None:
        status = "review"
    if any("\ufffd" in g.text or "\x00" in g.text or "(cid:" in g.text for g in glyphs):
        status = "review"
        corrections.append("unreliable_unicode")
    return {"paragraph_id": para.id, "page": line.page, "line_id": line.id,
            "paragraph_pages": sorted({l.page for l in para.lines}),
            "paragraph_text": " ".join(l.text for l in para.lines),
            "final_line": glyph_text(glyphs), "last_char": rightmost.text,
            "bbox": list(effective), "raw_bbox": list(line.bbox),
            "last_char_x1": effective[2], "threshold_x": threshold,
            "tail_ratio": (effective[2] - left) / (right - left) if right > left else None,
            "is_short": effective[2] < threshold, "status": status,
            "eligible": not para.math_ending, "end_reason": para.end_reason,
            "boundary_evidence": para.evidence, "body_bounds": bounds,
            "region": {"id": line.region_id, "label": line.region_label,
                       "score": line.region_score, "source": line.reason},
            "corrections": corrections,
            "context_before": para.lines[-2].text if len(para.lines) > 1 else ""}
