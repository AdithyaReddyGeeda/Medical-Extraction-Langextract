# -*- coding: utf-8 -*-
"""
Simple evaluation script for ClinicalExtract.

Computes rough precision/recall (and F1) against gold JSON annotations
in samples/ (optional .json alongside .txt with same base name).
Supports partial match (e.g. token overlap or normalized text match).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_gold(gold_path: Path) -> list[dict[str, Any]]:
    """Load gold extractions from a JSON file. Expected: list of {class, text, ...}."""
    with open(gold_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "extractions" in data:
        return data["extractions"]
    return []


def normalize_for_match(s: str) -> str:
    """Normalize text for comparison (lowercase, strip, collapse spaces)."""
    return " ".join(s.lower().strip().split())


def partial_match(pred_text: str, gold_text: str) -> bool:
    """True if normalized pred is contained in gold or vice versa (partial match)."""
    p = normalize_for_match(pred_text)
    g = normalize_for_match(gold_text)
    if not p or not g:
        return False
    return p in g or g in p


def exact_match(pred_text: str, gold_text: str) -> bool:
    """Exact normalized match."""
    return normalize_for_match(pred_text) == normalize_for_match(gold_text)


def compute_metrics(
    predicted: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    match_fn: str = "partial",
) -> dict[str, float]:
    """
    Compute precision, recall, F1 for extractions.
    predicted/gold: list of {"class": str, "text": str, ...}.
    match_fn: "partial" (overlap) or "exact".
    """
    match = partial_match if match_fn == "partial" else exact_match

    # By class (optional): aggregate over all classes for single P/R/F1
    tp = 0
    matched_gold = set()
    for p in predicted:
        p_class = p.get("class", "")
        p_text = p.get("text", "")
        for i, g in enumerate(gold):
            if i in matched_gold:
                continue
            if p_class != g.get("class", ""):
                continue
            if match(p_text, g.get("text", "")):
                tp += 1
                matched_gold.add(i)
                break

    num_pred = len(predicted)
    num_gold = len(gold)
    precision = tp / num_pred if num_pred else 0.0
    recall = tp / num_gold if num_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "pred": num_pred, "gold": num_gold}


def run_eval(samples_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    """
    Run evaluation over samples/ that have both .txt and .json (gold).
    Returns aggregate metrics and per-file results.
    """
    samples_dir = Path(samples_dir)
    results = []
    all_pred: list[dict] = []
    all_gold: list[dict] = []

    for txt_path in sorted(samples_dir.glob("*.txt")):
        gold_path = txt_path.with_suffix(".json")
        if not gold_path.exists():
            continue
        gold = load_gold(gold_path)
        # In a full pipeline we would run extractor on txt_path and get predicted.
        # Here we assume predicted are already in a companion _pred.json or we skip.
        pred_path = txt_path.with_name(txt_path.stem + "_pred.json")
        if not pred_path.exists():
            continue
        with open(pred_path, "r", encoding="utf-8") as f:
            pred = json.load(f)
        if isinstance(pred, list):
            pred_list = pred
        elif isinstance(pred, dict) and "extractions" in pred:
            pred_list = pred["extractions"]
        else:
            pred_list = []
        metrics = compute_metrics(pred_list, gold, match_fn="partial")
        results.append({"file": txt_path.name, "metrics": metrics})
        all_pred.extend(pred_list)
        all_gold.extend(gold)

    aggregate = compute_metrics(all_pred, all_gold, match_fn="partial") if all_pred and all_gold else {}
    report = {"per_file": results, "aggregate": aggregate}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "eval_results.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ClinicalExtract on samples with gold JSON.")
    parser.add_argument("--samples", default="samples", help="Directory containing .txt and .json gold files")
    parser.add_argument("--output", default=None, help="Directory to write eval_results.json")
    args = parser.parse_args()
    report = run_eval(args.samples, args.output)
    print("Aggregate:", report.get("aggregate", {}))
    if args.output:
        print(f"Wrote {Path(args.output) / 'eval_results.json'}")


if __name__ == "__main__":
    main()
