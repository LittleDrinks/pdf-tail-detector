import re
from pathlib import Path

import pymupdf
import pytest
from reportlab.pdfgen.canvas import Canvas

from src import scan_pdf
from src.layout import Glyph, PageData, TextLine
from src.paragraphs import refine_page


def _line(text: str, top: float, fontname: str = "NimbusRomNo9L-Regu") -> TextLine:
    glyphs = [Glyph(char, (100 + i * 5, top, 105 + i * 5, top + 10),
                    (100 + i * 5, top + 8), 10, fontname, text) for i, char in enumerate(text)]
    return TextLine(2, text, glyphs, text, (100, top, 100 + len(text) * 5, top + 10), top + 8, 10)


def _refine(lines):
    page = PageData(2, 612, 792, 0, (0, 0, 612, 792), lines)
    refine_page(page, 10, False)
    return page.lines


def test_algorithm_environment_is_removed_as_one_block():
    lines = [
        _line("Algorithm 1 Calibrated correction", 100),
        _line("Require: candidate actions", 110),
        _line("1: for each state do", 120),
        _line("4: end for", 130),
        _line("5: Sort the distinct values", 140),
        _line("6: for thresholds do", 150),
        _line("10: end if", 160),
        _line("11: end for", 170),
        _line("A normal paragraph resumes here.", 200),
    ]

    remaining = [line for line in _refine(lines) if line.kind != "excluded"]

    assert [line.text for line in remaining] == ["A normal paragraph resumes here."]


def test_math_fragment_is_candidate_safe_without_being_structural():
    fragment = _line("s", 100, "IFMQPP+CMMI5")

    assert _refine([fragment])[0].kind == "math"


def test_annotation_label_is_structural_metadata():
    annotation = _line("tail x=338.7", 100, "Helvetica")

    assert _refine([annotation])[0].reason == "legacy_annotation"


def test_margin_line_numbers_do_not_split_tail_and_split_references_stop_scan(tmp_path):
    path = tmp_path / "numbered.pdf"
    canvas = Canvas(str(path), pagesize=(612, 792))
    canvas.setFont("Helvetica", 10)
    canvas.drawString(108, 740, "ABSTRACT")
    canvas.drawString(108, 720, "A previous paragraph introduces the result and ends here.")
    canvas.drawString(108, 703, "A measured effect has a long explanation on its first line and")
    canvas.drawString(108, 691, "full-model timing controls.")
    canvas.setFont("Helvetica", 8)
    canvas.drawString(72, 700, "1")
    canvas.drawString(72, 688, "2")
    canvas.showPage()
    canvas.setFont("Helvetica", 12)
    canvas.drawString(108, 700, "R")
    canvas.drawString(117, 698, "EFERENCES")
    canvas.setFont("Helvetica", 10)
    canvas.drawString(108, 660, "A citation with enough words to look like a prose paragraph")
    canvas.drawString(108, 648, "but it belongs to the references list.")
    canvas.save()

    findings = scan_pdf(path)

    assert all(finding.page == 1 for finding in findings)
    tail = next(finding for finding in findings if finding.final_line == "full-model timing controls.")
    assert tail.paragraph_text.startswith("A measured effect")


@pytest.mark.parametrize(
    ("filename", "algorithm_page", "tail_anchors"),
    [
        (
            "what-does-automatic-differentiation-compute.pdf",
            7,
            ("fixed Ψ", "networks although they contain"),
        ),
        (
            "efficiently-computing-similarities.pdf",
            16,
            ("without privacy loss", "private datasets in the box",
             "data structures and are faster in the large d regime."),
        ),
    ],
)
def test_iclr_papers_keep_prose_tails_and_drop_algorithm_steps(filename, algorithm_page, tail_anchors):
    path = Path(__file__).parent / "fixtures" / "iclr" / filename
    with pymupdf.open(path) as pdf:
        assert re.search(r"Algorithm\s*1\b", pdf[algorithm_page - 1].get_text())

    findings = scan_pdf(path)

    assert findings
    assert not any(re.match(r"^\s*\d+\s*:", finding.final_line) for finding in findings)
    assert not any(finding.page == 1 and any(token in finding.final_line for token in ("University", "Research", "@")) for finding in findings)
    assert not any(re.match(r"^\([a-z]\)\s", finding.final_line) for finding in findings)
    for anchor in tail_anchors:
        assert any(anchor in finding.final_line for finding in findings)


def test_scan_pdf_keeps_the_original_bbox_order(tmp_path):
    from src.pipeline import analyze_pdf
    path = tmp_path / "compat.pdf"
    canvas = Canvas(str(path), pagesize=(612, 792))
    canvas.setFont("Times-Roman", 10)
    canvas.drawString(108, 692, "This paragraph introduces a complete ordinary explanation with enough words")
    canvas.drawString(108, 680, "and ends here.")
    canvas.save()
    item = analyze_pdf(path, backend="none", page_ratio=2 / 3)["candidates"][0]
    finding = scan_pdf(path)[0]
    x0, y0, x1, y1 = item["bbox"]
    assert finding.bbox == (x0, x1, y0, y1)
    assert finding.threshold_x == 408
    assert finding.final_line == item["final_line"]
