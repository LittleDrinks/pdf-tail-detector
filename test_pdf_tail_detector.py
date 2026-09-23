import re
from pathlib import Path

import pdfplumber
import pytest
from reportlab.pdfgen.canvas import Canvas

from pdf_tail_detector import Line, _is_display_math_line, _looks_structural, _without_algorithm_blocks, scan_pdf


class _Page:
    width = 612
    height = 792


def _char(text: str, fontname: str = "NimbusRomNo9L-Regu") -> dict:
    return {"text": text, "x0": 0, "x1": 5, "top": 0, "bottom": 10, "size": 10, "fontname": fontname}


def _line(text: str, top: float, fontname: str = "NimbusRomNo9L-Regu") -> Line:
    chars = [_char(char, fontname) for char in text]
    return Line(chars, text, 100, 100 + max(5, len(text) * 5), top, top + 10, "left", 10)


def test_algorithm_environment_is_removed_as_one_block():
    lines = [
        _line("Algorithm 1 Calibrated correction", 0),
        _line("Require: candidate actions", 10),
        _line("1: for each state do", 20),
        _line("4: end for", 30),
        _line("5: Sort the distinct values", 40),
        _line("6: for thresholds do", 50),
        _line("10: end if", 60),
        _line("11: end for", 70),
        _line("A normal paragraph resumes here.", 100),
    ]

    remaining = _without_algorithm_blocks(lines)

    assert [line.text for line in remaining] == ["A normal paragraph resumes here."]


def test_math_fragment_is_candidate_safe_without_being_structural():
    fragment = _line("s", 100, "IFMQPP+CMMI5")

    assert _is_display_math_line(fragment, _Page(), 10)
    assert not _looks_structural(fragment, _Page(), 10)


def test_annotation_label_is_structural_metadata():
    annotation = _line("tail x=338.7", 100, "Helvetica")

    assert _looks_structural(annotation, _Page(), 10)


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
            ("without privacy loss", "private datasets in the box"),
        ),
    ],
)
def test_iclr_papers_keep_prose_tails_and_drop_algorithm_steps(filename, algorithm_page, tail_anchors):
    path = Path(__file__).parent / "tests" / "fixtures" / "iclr" / filename
    with pdfplumber.open(path) as pdf:
        assert re.search(r"Algorithm\s*1\b", pdf.pages[algorithm_page - 1].extract_text() or "")

    findings = scan_pdf(path)

    assert findings
    assert not any(re.match(r"^\s*\d+\s*:", finding.final_line) for finding in findings)
    assert not any(finding.page == 1 and any(token in finding.final_line for token in ("University", "Research", "@")) for finding in findings)
    assert not any(re.match(r"^\([a-z]\)\s", finding.final_line) for finding in findings)
    for anchor in tail_anchors:
        assert any(anchor in finding.final_line for finding in findings)
