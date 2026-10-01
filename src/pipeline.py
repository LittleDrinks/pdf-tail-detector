"""PDF-only analysis, evidence reports, and annotation output."""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pymupdf

from .paragraphs import (body_size, document_bounds, measure, multi_column, normalized,
                        paragraph_bounds, recover_paragraphs, refine_page)
from .layout import PPDocLayout, extract_page, render_page, sha256


def references_y(lines) -> float | None:
    ordered = sorted(lines, key=lambda line: (line.baseline, line.bbox[0]))
    for i, line in enumerate(ordered):
        if normalized(line.text) == "references":
            return line.bbox[1]
        if i + 1 < len(ordered) and abs(ordered[i + 1].baseline - line.baseline) <= 5:
            if normalized(line.text + ordered[i + 1].text) == "references":
                return min(line.bbox[1], ordered[i + 1].bbox[1])
    return None


def analyze_pdf(input_pdf: Path, *, model_dir: Path = Path("outputs/models/PP-DocLayout-M"),
                backend: str = "onnx", tail_ratio: float = .75, page_ratio: float | None = None,
                threads: int = 2, dpi: float = 150, region_score: float = .5,
                max_pages: int | None = None, progress=None) -> dict:
    input_pdf = Path(input_pdf)
    for name, value in (("tail_ratio", tail_ratio), ("region_score", region_score)):
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{name} must be finite and between 0 and 1")
    if page_ratio is not None and (not math.isfinite(page_ratio) or not 0 < page_ratio < 1):
        raise ValueError("page_ratio must be finite and between 0 and 1")
    if not isinstance(threads, int) or threads < 1 or not math.isfinite(dpi) or not 72 <= dpi <= 300:
        raise ValueError("threads must be positive; dpi must be between 72 and 300")
    if max_pages is not None and max_pages < 1:
        raise ValueError("max_pages must be positive")
    started = time.perf_counter()
    model = PPDocLayout(model_dir, threads, backend) if backend != "none" else None
    loaded = time.perf_counter() - started
    pages = []
    references_started = False
    with pymupdf.open(input_pdf) as pdf:
        if pdf.needs_pass:
            raise ValueError("Encrypted PDF requires a decrypted copy.")
        total_pages = len(pdf)
        if not total_pages:
            raise ValueError("PDF has no pages.")
        for index, page in enumerate(pdf):
            if references_started or (max_pages and index >= max_pages):
                break
            tick = time.perf_counter()
            data = extract_page(page)
            data.timings["text_seconds"] = time.perf_counter() - tick
            cutoff = references_y(data.lines)
            references_started = cutoff is not None
            if cutoff is not None:
                for line in data.lines:
                    if line.bbox[1] >= cutoff - .1:
                        line.kind, line.reason = "excluded", "references_and_appendix"
            if not data.lines:
                data.status = "no_text_layer" if page.get_images() else "blank"
            elif model:
                tick = time.perf_counter()
                rgb, data.geometry = render_page(page, dpi)
                data.timings["render_seconds"] = time.perf_counter() - tick
                tick = time.perf_counter()
                data.regions = model.predict(rgb, data.geometry, index + 1, region_score)
                data.timings["model_seconds"] = time.perf_counter() - tick
                del rgb
                from .layout import associate
                # Preserve the document scope boundary even if the model misses References.
                active = [line for line in data.lines if line.reason != "references_and_appendix"]
                associate(active, data.regions)
            pages.append(data)
            if progress:
                progress(f"page {index + 1}: {len(data.regions)} regions")
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
    candidates = [item for item in measured if item["eligible"] and item["is_short"] and item["status"] == "checked"]
    review = [item for item in measured if item["eligible"] and item["status"] == "review"]
    unsupported = [page.number for page in pages if page.status not in {"checked", "blank"}]
    partial = bool(unsupported or review or (max_pages and len(pages) < total_pages and not references_started))
    evidence_pages = []
    for page in pages:
        evidence_pages.append({"page": page.number, "size": [page.width, page.height],
                               "rotation": page.rotation, "cropbox": list(page.cropbox),
                               "status": page.status, "geometry": page.geometry,
                               "timings": page.timings, "regions": [asdict(r) for r in page.regions],
                               "lines": [{"id": line.id, "text": line.text, "bbox": list(line.bbox),
                                          "baseline": line.baseline, "kind": line.kind, "reason": line.reason,
                                          "region_id": line.region_id, "repairs": line.repairs}
                                         for line in page.lines]})
    eligible = [item for item in measured if item["eligible"]]
    return {"schema_version": 2, "coordinate_system": "unrotated-crop-relative-xyxy",
            "input": {"path": str(input_pdf.resolve()), "sha256": sha256(input_pdf), "pages": total_pages},
            "status": "partial" if partial else "complete", "scope": "single-column English ICLR-like text PDF, main matter",
            "config": {"tail_ratio": tail_ratio, "page_ratio": page_ratio, "dpi": dpi,
                       "region_score": region_score, "threads": threads, "max_pages": max_pages,
                       "rule_version": "pdf-body-tail-v1", "body_size": size},
            "model": model.metadata() if model else {"backend": "none", "name": "explicit_geometry_baseline"},
            "coverage": {"pages_processed": len(pages), "pages_in_document": total_pages,
                         "stopped_at_references": references_started, "unsupported_pages": unsupported,
                         "eligible_paragraphs": len(eligible), "checked_paragraphs": len(eligible) - len(review),
                         "review_paragraphs": len(review),
                         "checked_fraction": (len(eligible) - len(review)) / len(eligible) if eligible else None,
                         "body_lines": sum(l.kind == "body" for p in pages for l in p.lines),
                         "unknown_lines": sum(l.kind == "unknown" for p in pages for l in p.lines)},
            "timings": {"model_load_seconds": loaded, "total_seconds": time.perf_counter() - started,
                        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform != "darwin" else 1024 ** 2)},
            "candidates": candidates, "review_items": review, "paragraphs": measured, "pages": evidence_pages}


def write_annotated(input_pdf: Path, output_pdf: Path, report: dict, diagnostic: bool = False) -> None:
    if Path(input_pdf).resolve() == Path(output_pdf).resolve():
        raise ValueError("Output PDF must differ from input PDF.")
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(input_pdf) as pdf:
        # Replace this tool's annotations on rerun without introducing text into the PDF layer.
        for page in pdf:
            for annotation in list(page.annots() or []):
                if annotation.info.get("title") == "PDF Tail Detector":
                    page.delete_annot(annotation)
        for item in report["candidates"] + report["review_items"]:
            page = pdf[item["page"] - 1]
            annotation = page.add_rect_annot(pymupdf.Rect(item["bbox"]))
            color = (1, 0, 0) if item["status"] == "checked" else (.7, .35, 1)
            annotation.set_colors(stroke=color)
            annotation.set_border(width=1)
            annotation.set_info(title="PDF Tail Detector", content=json.dumps({key: item[key] for key in
                                ("paragraph_id", "final_line", "tail_ratio", "threshold_x", "status", "end_reason")}, ensure_ascii=False))
            annotation.update()
            x = item["threshold_x"]
            y0, y1 = item["bbox"][1], item["bbox"][3]
            threshold = page.add_line_annot((x, y0 - 2), (x, y1 + 2))
            threshold.set_colors(stroke=(1, .55, 0))
            threshold.set_info(title="PDF Tail Detector", content="short-tail threshold")
            threshold.update()
        if diagnostic:
            for data in report["pages"]:
                page = pdf[data["page"] - 1]
                for region in data["regions"]:
                    annotation = page.add_rect_annot(pymupdf.Rect(region["bbox"]))
                    annotation.set_colors(stroke=(0, .65, .25) if region["label"] in {"text", "abstract", "content"}
                                          else (0, .4, 1) if region["label"].startswith("formula") else (.6, .6, .6))
                    annotation.set_opacity(.65)
                    annotation.set_info(title="PDF Tail Detector", content=f'{region["id"]}: {region["label"]} {region["score"]:.3f}')
                    annotation.update()
        pdf.save(output_pdf, garbage=3, deflate=True)


def cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description="PDF-only short-tail detection with PP-DocLayout-M on CPU.")
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/models/PP-DocLayout-M"))
    parser.add_argument("--backend", choices=("onnx", "paddle", "none"), default="onnx")
    rule = parser.add_mutually_exclusive_group()
    rule.add_argument("--tail-ratio", type=float, default=.75, help="fraction of body width, default .75")
    rule.add_argument("--ratio", type=float, help="compatibility rule: fraction of full page width")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dpi", type=float, default=150)
    parser.add_argument("--region-score", type=float, default=.5)
    parser.add_argument("--max-pages", type=int, help="explicit partial scan for diagnostics")
    parser.add_argument("--diagnostic", action="store_true", help="also annotate all model regions")
    args = parser.parse_args(argv)
    output = args.output or Path("outputs") / f"{args.input_pdf.stem}.tail-marked.pdf"
    json_path = args.json_path or output.with_suffix(".json")
    resolved = [args.input_pdf.resolve(), output.resolve(), json_path.resolve()]
    if len(set(resolved)) != 3:
        parser.error("Input PDF, output PDF and JSON report must have different paths.")
    try:
        report = analyze_pdf(args.input_pdf, model_dir=args.model_dir, backend=args.backend,
                             tail_ratio=args.tail_ratio, page_ratio=args.ratio, threads=args.threads,
                             dpi=args.dpi, region_score=args.region_score, max_pages=args.max_pages,
                             progress=lambda text: print(text, file=sys.stderr, flush=True))
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
