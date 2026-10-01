"""Recover paragraph boundaries from a stream of prose and structural events."""
from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field

from .layout import PageData, TextLine, enclosing, glyph_text, merge_visual_lines

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


def _heading(text: str) -> bool:
    named = normalized(text) in {"abstract", "references", "acknowledgments", "acknowledgements", "introduction"}
    numbered = (re.match(r"^(?:\d+(?:\.\d+)*\s+|[A-Z]\s+)", text) and len(text) < 120
                and sum(c.isupper() for c in text) > sum(c.islower() for c in text))
    return bool(named or numbered)


def _exclusion_reason(line: TextLine, page: PageData, size: float, words: list[str], math_ratio: float) -> str | None:
    text = line.text.strip()
    margin_digit = bool(re.fullmatch(r"\d+", text) and
                        (line.bbox[0] < page.width * .15 or line.bbox[1] > page.height * .92))
    if margin_digit or line.bbox[1] < page.height * .065 or line.bbox[3] > page.height * .95:
        return "margin_or_page_number"
    if text.lower().startswith("tail x="):
        return "legacy_annotation"
    if _heading(text):
        return "section_heading"
    return _environment_reason(line, size, words, math_ratio)


def _environment_reason(line: TextLine, size: float, words: list[str], math_ratio: float) -> str | None:
    if line.size < size * .92 and math_ratio < .2 and len(words) >= 3:
        return "small_prose_footnote"
    if re.match(r"^(?:Algorithm\s+\d|\d+\s*:|Require:|Ensure:|Input:|Output:)", line.text.strip(), re.I):
        return "algorithm_cue"
    if re.match(r"^[•●◦]\s*|^\([a-z]\)\s+", line.text.strip()):
        return "list_item"
    return None


def _resolve_formula_conflict(line: TextLine, words: list[str], math_ratio: float) -> None:
    if line.kind == "body" and math_ratio > .6 and len(words) < 2 and len(line.text.strip()) < 25:
        line.kind, line.reason = "math", "inline_math_fragment"
    if line.kind == "math" and len(words) >= 4 and math_ratio < .45:
        line.kind, line.reason = "body", "prose_overlaps_formula"
        line.repairs.append("formula_region_conflict")


def _classify_uncovered(line: TextLine, size: float, words: list[str], math_ratio: float, model: bool) -> None:
    if line.kind != "unknown":
        return
    text = line.text.strip()
    if math_ratio > .6 or (len(words) < 2 and any(c in text for c in "=∑∫≤≥")):
        line.kind, line.reason = "math", "math_geometry"
    elif (len(words) >= 3 or (len(words) >= 2 and len(text) >= 12)) and line.size >= size * .9:
        line.kind = "body"
        line.reason = "heuristic_uncovered_prose" if model else "geometry_baseline_prose"


def _refine_line(line: TextLine, page: PageData, size: float, model: bool) -> None:
    words = re.findall(r"[A-Za-z]{3,}", line.text.strip())
    math_ratio = sum(bool(MATH_FONT.search(g.font)) for g in line.glyphs) / len(line.glyphs)
    _resolve_formula_conflict(line, words, math_ratio)
    if line.kind == "excluded":
        return
    reason = _exclusion_reason(line, page, size, words, math_ratio)
    if reason:
        line.kind, line.reason = "excluded", reason
    else:
        _classify_uncovered(line, size, words, math_ratio, model)


def _attach_short_prose(page: PageData, size: float) -> None:
    for line in page.lines:
        if line.kind != "unknown" or line.size < size * .9:
            continue
        neighbors = [other for other in page.lines if other.kind == "body"
                     and abs(other.bbox[0] - line.bbox[0]) < 3
                     and 0 < line.baseline - other.baseline <= size * 1.5]
        if neighbors and any(c.isalpha() for c in line.text):
            line.kind, line.reason = "body", "aligned_short_prose"


def _exclude_continuations(page: PageData, size: float) -> None:
    # Supplemental ranges end at a visible gap, not at 'end for'.
    previous = None
    active = set()
    for line in sorted(page.lines, key=lambda line: (line.baseline, line.bbox[0])):
        gap = line.baseline - previous.baseline if previous else 0
        if gap > size * 1.65 and line.reason not in {"list_item", "algorithm_cue"}:
            active.clear()
        _update_environment(active, line)
        previous = line


def _update_environment(active: set, line: TextLine) -> None:
    if line.reason == "list_item":
        active.add("list")
    if re.match(r"^Algorithm\s+\d", line.text, re.I):
        active.add("algorithm")
    if active and line.kind != "excluded":
        # List takes precedence when the two supplemental cues overlap.
        line.kind, line.reason = "excluded", "list_continuation" if "list" in active else "algorithm_continuation"


def _exclude_frontmatter(page: PageData) -> None:
    if page.number != 1:
        return
    abstract = next((line for line in page.lines if normalized(line.text) == "abstract"), None)
    for line in (line for line in page.lines if abstract and line.baseline < abstract.baseline):
        line.kind, line.reason = "excluded", "front_matter"


def refine_page(page: PageData, size: float, model: bool) -> None:
    for line in page.lines:
        _refine_line(line, page, size, model)
    _attach_short_prose(page, size)
    _exclude_continuations(page, size)
    page.lines = merge_visual_lines(page.lines, size)
    _exclude_frontmatter(page)


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


def _local_bounds(page, prose, region, abstract, fallback):
    local = [line for line in page.lines if line.kind == "body"
             and (line.region_id == region.id if region else line in prose)]
    if len(local) < 2:
        return fallback
    return (min(line.bbox[0] for line in local), max(line.bbox[2] for line in local),
            len(local), "abstract_alignment" if abstract else "wrap_alignment")


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
        left, right, support, source = _local_bounds(page, prose, region, abstract, (left, right, support, source))
    if right <= left or support < 2:
        long = [line for line in prose if len(line.text) >= 45]
        if len(long) >= 2:
            left = min(line.bbox[0] for line in long)
            right = max(line.bbox[2] for line in long)
            support, source = len(long), "paragraph_alignment"
    return {"left": left, "right": right, "support": support, "source": source,
            "reliable": right > left and support >= 2}


class _ParagraphStream:
    """Accumulate prose while retaining math and boundary evidence."""

    def __init__(self, size: float):
        self.size = size
        self.paragraphs: list[Paragraph] = []
        self.current: list[TextLine] = []
        self.last_event: TextLine | None = None
        self.math_ending = False
        self.evidence: list[str] = []
        self.previous_page: PageData | None = None

    def finish(self, reason: str, status: str = "checked") -> None:
        if self.current:
            self.paragraphs.append(Paragraph(f"para-{len(self.paragraphs) + 1}", self.current,
                                             reason, status, self.math_ending, self.evidence))
        self.current, self.evidence, self.math_ending = [], [], False

    def page_boundary(self, page: PageData) -> None:
        first = next((line for line in page.lines if line.kind == "body"), None)
        before = page.lines[:page.lines.index(first)] if first else page.lines
        blocking = any(line.reason == "section_heading" or line.region_label == "paragraph_title" for line in before)
        last = self.current[-1]
        compatible = bool(first and abs(first.bbox[0] - last.bbox[0]) < self.size * 1.5
                          and abs(first.size - last.size) < self.size * .15)
        lower = bool(first and re.match(r"^[a-z(]", first.text))
        unfinished = not re.search(r"[.!?]\s*[)\]”\"']?$", last.text)
        if page.number == self.previous_page.number + 1 and compatible and not blocking and (unfinished or lower):
            self.evidence.append(f"cross_page:{self.previous_page.number}->{page.number}")
        elif blocking or not unfinished:
            self.finish("page_boundary_complete")
        else:
            self.finish("ambiguous_page_continuation", "review")
        self.last_event = None

    def ignored_exclusion(self, line: TextLine) -> bool:
        peripheral = (line.reason in {"margin_or_page_number", "small_prose_footnote"}
                      or line.region_label in {"header", "footer", "footnote", "number"})
        alongside = bool(self.current and line.bbox[0] > max(item.bbox[2] for item in self.current
                                                            if item.page == self.current[-1].page) + 8)
        return peripheral or alongside

    def excluded(self, line: TextLine) -> None:
        if self.ignored_exclusion(line):
            return
        if self.current:
            complete = bool(re.search(r"[.!?]\s*$", self.current[-1].text))
            self.finish("structural_boundary", "checked" if complete or self.math_ending else "review")
        self.last_event = line

    def unknown(self, line: TextLine) -> None:
        if self.current:
            self.evidence.append(f"unknown_fragment:{line.id}")
        self.last_event = line

    def math(self, line: TextLine) -> None:
        if self.current:
            self.math_ending = True
            self.evidence.append(f"display_math:p{line.page}:{line.id}")
        self.last_event = line

    def break_reason(self, line: TextLine, step: float) -> str | None:
        gap = line.baseline - self.last_event.baseline
        bold = any(re.search(r"bold|medi|cmbx|cmssbx", g.font, re.I) for g in line.glyphs[:12])
        entity = bool(ENTITY.match(line.text) and (bold or re.match(r"^Proof[.:]", line.text)))
        indent = (line.bbox[0] - min(item.bbox[0] for item in self.current) > self.size * .85
                  and len(re.findall(r"[A-Za-z]{3,}", line.text)) >= 4
                  and bool(re.search(r"[.!?]\s*$", self.current[-1].text)))
        normal_gap = gap > max(step * 1.35, step + self.size * .35)
        after_formula = self.last_event.kind == "math" and gap > step * 1.65
        if entity:
            return "entity_start"
        if (self.last_event.kind != "math" and (normal_gap or indent)) or after_formula:
            return "paragraph_spacing" if normal_gap else "indent_or_formula_gap"
        return None

    def body(self, line: TextLine, step: float) -> None:
        if self.current and self.last_event and self.last_event.page == line.page:
            reason = self.break_reason(line, step)
            if reason:
                self.finish(reason)
        self.current.append(line)
        self.math_ending = False
        self.last_event = line

    def consume_event(self, line: TextLine, step: float) -> None:
        if line.kind == "excluded":
            self.excluded(line)
        elif line.kind == "unknown":
            self.unknown(line)
        elif line.kind == "math":
            self.math(line)
        else:
            self.body(line, step)


    def consume(self, page: PageData) -> None:
        if page.status != "checked":
            self.finish("unsupported_page_boundary", "review")
            self.previous_page, self.last_event = page, None
            return
        if self.current and self.previous_page:
            self.page_boundary(page)
        step = leading(page.lines, self.size)
        for line in page.lines:
            self.consume_event(line, step)
        self.previous_page = page


def recover_paragraphs(pages: list[PageData], size: float) -> list[Paragraph]:
    stream = _ParagraphStream(size)
    for page in pages:
        stream.consume(page)
    stream.finish("document_end")
    return stream.paragraphs


def _effective_glyphs(para: Paragraph) -> tuple[list, list[str]]:
    line = para.lines[-1]
    glyphs, corrections = list(line.glyphs), list(line.repairs)
    ordered = sorted(glyphs, key=lambda g: g.bbox[0])
    if ordered and ordered[-1].text in QED and re.search(r"\bProof\b", " ".join(item.text for item in para.lines), re.I):
        marker = ordered[-1]
        others = [g for g in glyphs if g is not marker]
        if others and marker.bbox[0] - max(g.bbox[2] for g in others) > line.size * 1.8:
            glyphs = others
            corrections.append("removed_isolated_proof_qed")
    return glyphs, corrections


def measure(para: Paragraph, page: PageData, bounds: dict, ratio: float,
            page_ratio: float | None = None) -> dict:
    line = para.lines[-1]
    glyphs, corrections = _effective_glyphs(para)
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
            "paragraph_pages": sorted({item.page for item in para.lines}),
            "paragraph_text": " ".join(item.text for item in para.lines),
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
