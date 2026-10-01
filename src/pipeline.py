"""PDF-only analysis, evidence reports, and annotation output."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pymupdf

from .layout import PPDocLayout, associate, extract_page, render_page, sha256
from .paragraphs import (
    body_size,
    document_bounds,
    measure,
    multi_column,
    normalized,
    paragraph_bounds,
    recover_paragraphs,
    refine_page,
)


def _scope_headings(lines) -> list[tuple[float, bool]]:
    ordered = sorted(lines, key=lambda line: (line.baseline, line.bbox[0]))
    headings = []
    for index, line in enumerate(ordered):
        pair = ordered[index:index + 2]
        joined = " ".join(item.text for item in pair if abs(item.baseline - line.baseline) <= 5)
        reference = any(normalized(text) == "references" for text in (line.text, joined))
        appendix = any(re.match(r"^appendix\b", text, re.I)
                       or (text.isupper() and re.match(r"^[A-Z](?:\.\d+)*\s+[A-Z]", text))
                       for text in (line.text, joined))
        if reference or appendix:
            headings.append((line.bbox[1] - .1, reference))
    return headings


def _validate_ratios(tail_ratio, region_score, page_ratio) -> None:
    ratios = {"tail_ratio": tail_ratio, "region_score": region_score}
    if page_ratio is not None:
        ratios["page_ratio"] = page_ratio
    for name, value in ratios.items():
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be finite and between 0 and 1")


def _validate_runtime(threads, dpi, max_pages, backend) -> None:
    if not isinstance(threads, int) or threads < 1 or not math.isfinite(dpi) or not 72 <= dpi <= 300:
        raise ValueError("threads must be positive; dpi must be between 72 and 300")
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be positive")
    if backend not in {"onnx", "none"}:
        raise ValueError(f"Unknown backend: {backend}")


def _scope_page(data, references: bool) -> bool:
    headings = _scope_headings(data.lines)
    for line in data.lines:
        in_references = next((state for y, state in reversed(headings) if line.bbox[1] >= y), references)
        if in_references:
            line.kind, line.reason = "excluded", "references"
    return headings[-1][1] if headings else references


def _read_page(page, model, dpi, region_score, references):
    data = extract_page(page)
    references = _scope_page(data, references)
    active = [line for line in data.lines if line.reason != "references"]
    if not data.lines:
        data.status = "no_text_layer" if page.get_images() else "blank"
    elif model and active:
        rgb, data.geometry = render_page(page, dpi)
        data.regions = model.predict(rgb, data.geometry, page.number + 1, region_score)
        associate(active, data.regions)
    return data, references


def _read_pages(input_pdf, model, dpi, region_score, max_pages):
    pages, references, references_seen = [], False, False
    with pymupdf.open(input_pdf) as pdf:
        if pdf.needs_pass or not len(pdf):
            raise ValueError("PDF must contain pages and be decrypted.")
        total_pages = len(pdf)
        for index, page in enumerate(pdf):
            if max_pages and index >= max_pages:
                break
            data, references = _read_page(page, model, dpi, region_score, references)
            references_seen |= any(line.reason == "references" for line in data.lines)
            pages.append(data)
    return pages, total_pages, references_seen


def _page_evidence(page) -> dict:
    return {"page": page.number, "size": [page.width, page.height],
            "rotation": page.rotation, "cropbox": list(page.cropbox),
            "status": page.status, "geometry": page.geometry,
            "regions": [asdict(region) for region in page.regions],
            "lines": [{"id": line.id, "text": line.text, "bbox": list(line.bbox),
                       "baseline": line.baseline, "kind": line.kind, "reason": line.reason,
                       "region_id": line.region_id, "repairs": line.repairs} for line in page.lines]}


def analyze_pdf(input_pdf: Path, *, model_dir: Path = Path("outputs/cache/models/PP-DocLayout-M"),
                backend: str = "onnx", tail_ratio: float = .75, page_ratio: float | None = None,
                threads: int = 2, dpi: float = 150, region_score: float = .5,
                max_pages: int | None = None) -> dict:
    input_pdf = Path(input_pdf)
    _validate_ratios(tail_ratio, region_score, page_ratio)
    _validate_runtime(threads, dpi, max_pages, backend)
    started = time.perf_counter()
    model = PPDocLayout(model_dir, threads) if backend != "none" else None
    pages, total_pages, references_seen = _read_pages(input_pdf, model, dpi, region_score, max_pages)
    size = body_size(pages)
    for page in pages:
        refine_page(page, size, model is not None)
        if page.status == "checked" and multi_column(page):
            page.status = "unsupported_multi_column"
    bounds = document_bounds([page for page in pages if page.status == "checked"])
    paragraphs = recover_paragraphs(pages, size)
    by_page = {page.number: page for page in pages}
    measured = [measure(para, by_page[para.lines[-1].page],
                        paragraph_bounds(para, by_page[para.lines[-1].page], bounds),
                        tail_ratio, page_ratio) for para in paragraphs]
    config = {"tail_ratio": tail_ratio, "page_ratio": page_ratio, "dpi": dpi,
              "region_score": region_score, "threads": threads, "max_pages": max_pages,
              "rule_version": "pdf-body-tail-v2", "body_size": size, "body_bounds": list(bounds)}
    candidates = [item for item in measured if item["eligible"] and item["is_short"] and item["status"] == "checked"]
    review = [item for item in measured if item["eligible"] and item["status"] == "review"]
    unsupported = [page.number for page in pages if page.status not in {"checked", "blank"}]
    partial = bool(unsupported or review or (max_pages and len(pages) < total_pages))
    eligible = [item for item in measured if item["eligible"]]
    return {"schema_version": 2, "coordinate_system": "unrotated-crop-relative-xyxy",
            "input": {"path": str(input_pdf.resolve()), "sha256": sha256(input_pdf), "pages": total_pages},
            "status": "partial" if partial else "complete",
            "scope": "single-column English ICLR-like text PDF, main matter and appendices; references excluded",
            "config": config,
            "model": model.metadata() if model else {"backend": "none", "name": "explicit_geometry_baseline"},
            "coverage": {"pages_processed": len(pages), "pages_in_document": total_pages,
                         "references_excluded": references_seen, "unsupported_pages": unsupported,
                         "eligible_paragraphs": len(eligible), "checked_paragraphs": len(eligible) - len(review),
                         "review_paragraphs": len(review),
                         "checked_fraction": (len(eligible) - len(review)) / len(eligible) if eligible else None,
                         "body_lines": sum(line.kind == "body" for p in pages for line in p.lines),
                         "unknown_lines": sum(line.kind == "unknown" for p in pages for line in p.lines)},
            "timings": {"total_seconds": time.perf_counter() - started},
            "candidates": candidates, "review_items": review, "paragraphs": measured, "pages": [_page_evidence(page) for page in pages]}


ANNOTATION_TITLE = "PDF Tail Detector"


def _remove_annotations(page) -> None:
    for annotation in list(page.annots() or []):
        if annotation.info.get("title") == ANNOTATION_TITLE:
            page.delete_annot(annotation)


def _mark_tail(page, item) -> None:
    box = (pymupdf.Rect(item["bbox"]) + (-1.5, -1.5, 1.5, 1.5)) & (page.rect * page.derotation_matrix)
    color = (1, 0, 0) if item["status"] == "checked" else (.7, .35, 1)
    # Separate fill and outline so only the fill inherits the original 8% opacity.
    fill = page.add_rect_annot(box)
    fill.set_colors(stroke=color, fill=color)
    fill.set_border(width=0)
    fill.set_opacity(.08)
    fill.set_info(title=ANNOTATION_TITLE, content="short-tail fill")
    fill.update()
    annotation = page.add_rect_annot(box)
    annotation.set_colors(stroke=color)
    annotation.set_border(width=1)
    annotation.set_info(title=ANNOTATION_TITLE, content=json.dumps({key: item[key] for key in
                        ("paragraph_id", "final_line", "last_char_x1", "tail_ratio", "threshold_x", "status", "end_reason")}, ensure_ascii=False))
    annotation.update()


def _page_thresholds(page, report) -> set[float]:
    config = report["config"]
    if config["page_ratio"] is not None:
        return {(page.rect * page.derotation_matrix).width * config["page_ratio"]}
    left, right, support = config["body_bounds"]
    thresholds = {item["threshold_x"] for item in report["paragraphs"]
                  if item["page"] == page.number + 1 and item["body_bounds"]["reliable"]}
    if support >= 2:
        thresholds.add(left + config["tail_ratio"] * (right - left))
    return thresholds


def _mark_diagnostics(pdf, report) -> None:
    for data in report["pages"]:
        page = pdf[data["page"] - 1]
        for region in data["regions"]:
            annotation = page.add_rect_annot(pymupdf.Rect(region["bbox"]))
            color = ((0, .65, .25) if region["label"] in {"text", "abstract", "content"}
                     else (0, .4, 1) if region["label"].startswith("formula") else (.6, .6, .6))
            annotation.set_colors(stroke=color)
            annotation.set_opacity(.65)
            annotation.set_info(title=ANNOTATION_TITLE, content=f'{region["id"]}: {region["label"]} {region["score"]:.3f}')
            annotation.update()


def _mark_page(page, report) -> None:
    _remove_annotations(page)
    for x in sorted(_page_thresholds(page, report)):
        height = (page.rect * page.derotation_matrix).height
        annotation = page.add_line_annot((x, 0), (x, height))
        annotation.set_colors(stroke=(1, .55, 0))
        annotation.set_opacity(.8)
        annotation.set_border(width=1, dashes=[4, 3])
        annotation.set_info(title=ANNOTATION_TITLE, content="short-tail threshold")
        annotation.update()
    for item in report["candidates"] + report["review_items"]:
        if item["page"] == page.number + 1:
            _mark_tail(page, item)


def write_annotated(input_pdf: Path, output_pdf: Path, report: dict, diagnostic: bool = False) -> None:
    if Path(input_pdf).resolve() == Path(output_pdf).resolve():
        raise ValueError("Output PDF must differ from input PDF.")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(input_pdf) as pdf:
        for page in pdf:
            _mark_page(page, report)
        if diagnostic:
            _mark_diagnostics(pdf, report)
        pdf.save(output_pdf, garbage=3, deflate=True)


def cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PDF-only short-tail detection with PP-DocLayout-M on CPU.")
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/cache/models/PP-DocLayout-M"))
    parser.add_argument("--backend", choices=("onnx", "none"), default="onnx")
    rule = parser.add_mutually_exclusive_group()
    rule.add_argument("--tail-ratio", type=float, default=.75, help="fraction of body width, default .75")
    rule.add_argument("--ratio", type=float, help="compatibility rule: fraction of full page width")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dpi", type=float, default=150)
    parser.add_argument("--region-score", type=float, default=.5)
    parser.add_argument("--max-pages", type=int, help="explicit partial scan for diagnostics")
    parser.add_argument("--diagnostic", action="store_true", help="also annotate all model regions")
    args = parser.parse_args(argv)
    output = args.output or Path("outputs/results") / f"{args.input_pdf.stem}.tail-marked.pdf"
    json_path = args.json_path or output.with_suffix(".json")
    resolved = [args.input_pdf.resolve(), output.resolve(), json_path.resolve()]
    if len(set(resolved)) != 3:
        parser.error("Input PDF, output PDF and JSON report must have different paths.")
    try:
        report = analyze_pdf(args.input_pdf, model_dir=args.model_dir, backend=args.backend,
                             tail_ratio=args.tail_ratio, page_ratio=args.ratio, threads=args.threads,
                             dpi=args.dpi, region_score=args.region_score, max_pages=args.max_pages)
        write_annotated(args.input_pdf, output, report, args.diagnostic)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    except (ValueError, FileNotFoundError, RuntimeError, pymupdf.FileDataError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for item in report["candidates"]:
        print(f'p{item["page"]} {item["paragraph_id"]}: {item["final_line"]}\n'
              f'  X={item["last_char_x1"]:.2f}, threshold={item["threshold_x"]:.2f}, body ratio={item["tail_ratio"]}')
    print(f'{len(report["candidates"])} candidates; {len(report["review_items"])} review items; '
          f'status={report["status"]}; {report["timings"]["total_seconds"]:.2f}s')
    print(f"PDF: {output}\nJSON: {json_path}")
    # A partial report is useful output but distinct from a fully checked document.
    return 2 if report["status"] == "partial" else 0
