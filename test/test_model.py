"""Real-weight regressions; absent weights are explicitly skipped."""
from pathlib import Path

import pymupdf
import pytest

from src.pipeline import analyze_pdf, write_annotated

MODEL = Path(__file__).resolve().parents[1] / "outputs/cache/models/PP-DocLayout-M"
pytestmark = pytest.mark.skipif(not (MODEL / "model.onnx").exists(), reason="PP-DocLayout-M weights not prepared")


@pytest.mark.parametrize("filename,anchors", [
    ("efficiently-computing-similarities.pdf", ["computes an approximation to", "kernels.", "implemented in Python 3.9"]),
    ("what-does-automatic-differentiation-compute.pdf",
     ["fixed Ψ", "networks although they contain", "x if and only if f is continuously differentiable at f."]),
])
def test_real_prose_tails_survive_inline_math_and_excluded_regions(filename, anchors):
    report = analyze_pdf(Path(__file__).parent / "fixtures/iclr" / filename,
                         model_dir=MODEL, page_ratio=2/3)
    for anchor in anchors:
        assert any(anchor in item["final_line"] for item in report["candidates"])
    assert not any(item["final_line"].startswith("1:") for item in report["candidates"])
    assert not any("Throughout the paper" in item["final_line"] for item in report["candidates"])
    assert not any("Corollary 3.2 allows us to express" == item["final_line"] for item in report["candidates"])
    assert not any(item["final_line"] == "|X|" for item in report["candidates"])
    assert report["coverage"]["pages_processed"] == report["input"]["pages"]
    assert not any(item["page"] == 14 and "such that" in item["final_line"] for item in report["candidates"])


@pytest.mark.parametrize("filename", ["iclr/efficiently-computing-similarities.pdf",
                                     "iclr/what-does-automatic-differentiation-compute.pdf", "holdout/lora.pdf"])
def test_first_page_has_only_one_full_guide(tmp_path, filename):
    path = Path(__file__).parent / "fixtures" / filename
    report = analyze_pdf(path, model_dir=MODEL)
    output = tmp_path / "marked.pdf"
    write_annotated(path, output, report)
    with pymupdf.open(path) as before, pymupdf.open(output) as after:
        page = after[0]
        guides = [a.vertices for a in page.annots() if a.type[1] == "Line"]
        assert len(guides) == 1
        assert guides[0][0] == pytest.approx((405, 0), abs=.1)
        assert guides[0][1] == pytest.approx((405, page.rect.height), abs=.1)
        assert before[0].get_text() == page.get_text()
    rerun = tmp_path / "rerun.pdf"
    write_annotated(output, rerun, report)
    with pymupdf.open(rerun) as pdf:
        page = pdf[0]
        assert [a.vertices for a in page.annots() if a.type[1] == "Line"] == guides
