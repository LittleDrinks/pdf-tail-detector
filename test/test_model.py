"""Real-weight regressions; absent weights are explicitly skipped."""
from pathlib import Path
import pytest
from src import analyze_pdf

MODEL = Path(__file__).resolve().parents[1] / "outputs/models/PP-DocLayout-M"
pytestmark = pytest.mark.skipif(not (MODEL / "model.onnx").exists(), reason="PP-DocLayout-M weights not prepared")


@pytest.mark.parametrize("filename,anchors", [
    ("efficiently-computing-similarities.pdf", ["computes an approximation to", "kernels.", "implemented in Python 3.9"]),
    ("what-does-automatic-differentiation-compute.pdf", ["fixed Ψ", "networks although they contain"]),
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
