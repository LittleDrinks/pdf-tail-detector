from pathlib import Path

import pymupdf
import pytest
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from src.layout import (
    Glyph,
    PageData,
    Region,
    TextLine,
    associate,
    map_box,
    render_page,
)
from src.paragraphs import Paragraph, measure
from src.pipeline import analyze_pdf, write_annotated


def row(canvas, y, text, full=False, x=108):
    obj = canvas.beginText(x, 792 - y)
    obj.setFont("Times-Roman", 10)
    if full:
        obj.setHorizScale(396 / stringWidth(text, "Times-Roman", 10) * 100)
    obj.textOut(text)
    canvas.drawText(obj)


def test_body_width_formula_bridge_and_short_single_line(tmp_path):
    path = tmp_path / "body.pdf"
    canvas = Canvas(str(path), pagesize=(612, 792))
    row(canvas, 100, "The first paragraph provides a sufficiently long ordinary prose explanation.", full=True)
    row(canvas, 112, "Short tail.")
    row(canvas, 140, "A second ordinary paragraph has enough text to establish the body width.", full=True)
    row(canvas, 152, "This conclusion reaches the full measure and ends the second paragraph.", full=True)
    row(canvas, 180, "Our mathematical construction can be written in the following form", full=True)
    row(canvas, 192, "and is")
    row(canvas, 210, "x = y + 1", x=250)
    row(canvas, 222, "where the terms are positive.")
    row(canvas, 250, "We prove this.")
    row(canvas, 280, "References")
    row(canvas, 310, "An ordinary looking citation must never become a prose tail.")
    canvas.save()

    report = analyze_pdf(path, backend="none")
    ends = {item["final_line"]: item for item in report["candidates"]}
    assert "Short tail." in ends
    assert "where the terms are positive." in ends
    assert "We prove this." in ends
    assert "and is" not in ends
    assert "Our mathematical" in ends["where the terms are positive."]["paragraph_text"]
    assert ends["Short tail."]["threshold_x"] == pytest.approx(405, abs=.2)
    assert report["coverage"]["stopped_at_references"]
    assert not any("citation" in item["final_line"] for item in report["candidates"])


def test_cross_page_only_reports_the_real_end(tmp_path):
    path = tmp_path / "cross.pdf"
    canvas = Canvas(str(path), pagesize=(612, 792))
    row(canvas, 690, "Our argument begins here and gives a complete description of the construction", full=True)
    row(canvas, 702, "which continues and")
    canvas.showPage()
    row(canvas, 90, "uses the following values to establish the conclusion of this argument", full=True)
    row(canvas, 102, "and ends here.")
    canvas.save()
    report = analyze_pdf(path, backend="none")
    assert len(report["candidates"]) == 1
    finding = report["candidates"][0]
    assert finding["page"] == 2
    assert finding["paragraph_pages"] == [1, 2]
    assert finding["boundary_evidence"] == ["cross_page:1->2"]


@pytest.mark.parametrize("rotation,crop", [(0, False), (90, False), (270, True)])
def test_rotation_crop_mapping_and_annotation_does_not_change_text(tmp_path, rotation, crop):
    original = tmp_path / "base.pdf"
    canvas = Canvas(str(original), pagesize=(612, 792))
    row(canvas, 120, "A long line fills the text measure and establishes a reliable body boundary.", full=True)
    row(canvas, 132, "A short ending.")
    row(canvas, 160, "Another long line fills the text measure to support the boundary estimate.", full=True)
    row(canvas, 172, "A second short ending.")
    canvas.save()
    path = tmp_path / "geometry.pdf"
    with pymupdf.open(original) as pdf:
        if crop:
            pdf[0].set_cropbox(pymupdf.Rect(50, 50, 562, 742))
        pdf[0].set_rotation(rotation)
        note = pdf[0].add_text_annot((20, 20), "A reader's note")
        note.set_info(title="Reader")
        note.update()
        pdf.save(path)
    with pymupdf.open(path) as pdf:
        words = pdf[0].get_text("words")
        rgb, geometry = render_page(pdf[0], 123)
        box = words[0][:4]
        scale = geometry["scale"]
        assert map_box([v * scale for v in box], geometry) == pytest.approx(box)
        assert pdf[0].rotation == rotation
        assert rgb.shape[2] == 3
    report = analyze_pdf(path, backend="none")
    assert len(report["candidates"]) == 2
    output = tmp_path / "marked.pdf"
    write_annotated(path, output, report)
    with pymupdf.open(path) as before, pymupdf.open(output) as after:
        assert before[0].get_text() == after[0].get_text()
        assert before[0].rotation == after[0].rotation
        page = after[0]
        annotations = list(page.annots())
        guides = [item for item in annotations if item.type[1] == "Line"]
        assert len(guides) == 1
        guide = guides[0]
        assert guide.vertices[0] == pytest.approx((report["candidates"][0]["threshold_x"], 0))
        assert guide.vertices[1] == pytest.approx((report["candidates"][0]["threshold_x"], 692 if crop else 792))
        assert guide.border["dashes"] == (4, 3)
        assert guide.border["width"] == 1
        assert guide.opacity == pytest.approx(.8)
        assert guide.colors["stroke"] == pytest.approx((1, .55, 0))
        assert any(item.info["title"] == "Reader" for item in annotations)
        marked_rgb, _ = render_page(page, 123)
        assert (marked_rgb == rgb).all()
    repeated = analyze_pdf(output, backend="none")
    assert [f["bbox"] for f in repeated["candidates"]] == [f["bbox"] for f in report["candidates"]]
    rerun = tmp_path / "rerun.pdf"
    write_annotated(output, rerun, repeated)
    with pymupdf.open(rerun) as pdf:
        page = pdf[0]
        assert len(list(page.annots())) == len(annotations)


def test_page_ratio_guide_is_present_without_candidates(tmp_path):
    path = tmp_path / "blank.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page(width=612, height=792)
        pdf.new_page(width=612, height=792)
        pdf.save(path)
    report = analyze_pdf(path, backend="none", page_ratio=2 / 3)
    assert not report["candidates"]
    output = tmp_path / "marked.pdf"
    write_annotated(path, output, report)
    with pymupdf.open(output) as pdf:
        for page in pdf:
            guides = list(page.annots())
            assert len(guides) == 1
            assert guides[0].vertices == [(408, 0), (408, 792)]
            assert guides[0].border["dashes"] == (4, 3)


def test_qed_is_peripheral_only_in_proof_and_rightmost_extent_is_used():
    glyphs = [Glyph("Proof.", (108, 100, 140, 110), (108, 108), 10, "Times", "0:0"),
              Glyph("ends.", (142, 100, 180, 110), (142, 108), 10, "Times", "0:0"),
              Glyph("□", (496, 100, 504, 110), (496, 108), 10, "Symbol", "0:0")]
    line = TextLine(1, "0:0", glyphs, "Proof. ends. □", (108, 100, 504, 110), 108, 10, "body")
    page = PageData(1, 612, 792, 0, (0, 0, 612, 792), [line])
    bounds = {"left": 108, "right": 504, "support": 5, "source": "fixture", "reliable": True}
    result = measure(Paragraph("p", [line], "document_end"), page, bounds, .75)
    assert result["last_char_x1"] == 180
    assert result["is_short"]
    assert "removed_isolated_proof_qed" in result["corrections"]
    line.text = "An inline square is a mathematical symbol."
    assert not measure(Paragraph("p", [line], "document_end"), page, bounds, .75)["is_short"]
    line.glyphs = [Glyph("wide", (108, 100, 190, 110), (108, 108), 10, "Times", "0:0"),
                   Glyph("later", (170, 100, 180, 110), (170, 108), 10, "Times", "0:0")]
    assert measure(Paragraph("p", [line], "document_end"), page, bounds, .75)["last_char_x1"] == 190


def test_region_touch_does_not_exclude_prose():
    glyphs = [Glyph(str(i), (108 + i * 5, 100, 113 + i * 5, 110), (108 + i * 5, 108), 10, "Times", "0:0")
              for i in range(20)]
    line = TextLine(1, "0:0", glyphs, "ordinary prose", (108, 100, 208, 110), 108, 10)
    associate([line], [Region("table", "table", .9, (200, 100, 400, 300)),
                       Region("body", "text", .95, (100, 95, 504, 150))])
    assert line.kind == "body"
    assert line.region_id == "body"


def test_missing_model_never_silently_falls_back(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model missing"):
        analyze_pdf(tmp_path / "paper.pdf", model_dir=tmp_path / "absent")


def test_scanned_and_two_column_inputs_are_explicitly_unsupported(tmp_path):
    scanned = tmp_path / "scan.pdf"
    from PIL import Image
    image_path = tmp_path / "scan.png"
    Image.new("RGB", (600, 800), "white").save(image_path)
    pdf = pymupdf.open()
    page = pdf.new_page(width=612, height=792)
    page.insert_image(page.rect, filename=str(image_path))
    pdf.save(scanned)
    report = analyze_pdf(scanned, backend="none")
    assert report["status"] == "partial"
    assert report["pages"][0]["status"] == "no_text_layer"
    columns = tmp_path / "columns.pdf"
    canvas = Canvas(str(columns), pagesize=(612, 792))
    for y in (100, 112, 124, 136):
        row(canvas, y, "This prose belongs to the left column.", x=60)
        row(canvas, y, "This prose belongs to the right column.", x=330)
    canvas.save()
    report = analyze_pdf(columns, backend="none")
    assert report["status"] == "partial"
    assert report["pages"][0]["status"] == "unsupported_multi_column"


@pytest.mark.parametrize("name,value", [("tail_ratio", float("nan")), ("page_ratio", float("inf")), ("dpi", 0), ("threads", 0)])
def test_invalid_configuration_fails_before_scan(name, value):
    with pytest.raises(ValueError):
        analyze_pdf(Path("unused.pdf"), backend="none", **{name: value})
