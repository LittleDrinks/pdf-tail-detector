#!/usr/bin/env python3
"""One-to-one endpoint scoring against detector-independent frozen labels."""
from __future__ import annotations
import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.layout import sha256


def close(pred: dict, label: dict) -> bool:
    a, b = pred["bbox"], label["bbox"]
    return (pred["page"] == label["page"] and abs((a[1] + a[3] - b[1] - b[3]) / 2) <= 4
            and min(a[2], b[2]) > max(a[0], b[0]))


def evaluate(labels: dict, predictions: list[dict], policy: str, review: list[dict] = ()) -> dict:
    endpoints = labels["endings"]
    certain = [row for row in endpoints if row["eligible"] and not row["uncertain"]]
    positive = [row for row in certain if row[f"is_short_{policy}"]]
    uncertain = [row for row in endpoints if row["uncertain"]]
    predictions = [row for row in predictions if row["page"] in labels["pages"]]
    available = list(positive)
    tp, fp, ignored = [], [], []
    for prediction in predictions:
        matching = [row for row in available if close(prediction, row)]
        if matching:
            match = min(matching, key=lambda row: abs(row["bbox"][1] - prediction["bbox"][1]))
            available.remove(match)
            tp.append({"label_id": match["id"], "prediction": prediction})
        elif any(close(prediction, row) for row in uncertain):
            ignored.append(prediction)
        else:
            fp.append(prediction)
    review_hits = [row["id"] for row in available if any(close(pred, row) for pred in review)]
    nonbody = [pred for pred in fp if not any(close(pred, row) for row in certain)]
    precision = len(tp) / (len(tp) + len(fp)) if tp or fp else None
    recall = len(tp) / len(positive) if positive else None
    return {"policy": policy, "pages": len(labels["pages"]), "eligible_ends": len(certain),
            "true_short_ends": len(positive), "uncertain_ends": len(uncertain),
            "tp": len(tp), "fp": len(fp), "fn": len(available),
            "precision": precision, "recall": recall,
            "f1": 2 * len(tp) / (2 * len(tp) + len(fp) + len(available)) if tp or fp or available else None,
            "nonbody_or_false_boundary_fp": len(nonbody), "false_positives_per_page": len(fp) / len(labels["pages"]),
            "review_true_short_ends": len(review_hits), "review_short_ids": review_hits,
            "uncertain_predictions_excluded_from_precision": len(ignored),
            "matches": tp, "false_positives": fp, "false_negatives": available,
            "ignored_uncertain_predictions": ignored}


def baseline_predictions(path: Path) -> list[dict]:
    spec = importlib.util.spec_from_file_location("frozen_baseline", ROOT / "outputs/verification/baseline/pdf_tail_detector.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    result = []
    for finding in module.scan_pdf(path):
        row = asdict(finding)
        x0, x1, y0, y1 = row["bbox"]
        row["bbox"] = [x0, y0, x1, y1]
        result.append(row)
    return result


def aggregate(results: list[dict]) -> dict:
    summed = {key: sum(r[key] for r in results) for key in
              ("pages", "eligible_ends", "true_short_ends", "uncertain_ends", "tp", "fp", "fn",
               "nonbody_or_false_boundary_fp", "review_true_short_ends")}
    tp, fp, fn = summed["tp"], summed["fp"], summed["fn"]
    return {**summed, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--reports", type=Path, nargs="*")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--policy", choices=("page", "body"), default="page")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.labels.read_text())
    reports = [json.loads(path.read_text()) for path in args.reports or []]
    results = []
    for document in data["documents"]:
        path = ROOT / document["path"]
        if sha256(path) != document["sha256"]:
            raise ValueError(f"Fixture changed: {path}")
        if args.baseline:
            if args.policy != "page":
                raise ValueError("Frozen baseline only implements page-width policy")
            predictions, review = baseline_predictions(path), []
        else:
            report = next(row for row in reports if row["input"]["sha256"] == document["sha256"])
            predictions, review = report["candidates"], report["review_items"]
        results.append({"document": document["path"], **evaluate(document, predictions, args.policy, review)})
    output = {"split": data["split"], "labels_sha256": sha256(args.labels), "baseline": args.baseline,
              "policy": args.policy, "aggregate": aggregate(results), "documents": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output["aggregate"], indent=2))
