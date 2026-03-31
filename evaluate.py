# -*- coding: utf-8 -*-
"""
Offline evaluation: run extractor on samples with gold JSON, score, report.

Usage:
  python evaluate.py --provider gemini --model gemini-2.5-flash
  python evaluate.py --skip-extraction --match partial
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from extractor import extract, extractions_to_serializable
from utils.eval import compute_metrics, compute_per_class_metrics, load_gold

load_dotenv()


def _extract_kwargs_for_provider(provider: str, model_id: str) -> dict[str, Any]:
    """Map provider choice to extractor.extract() keyword arguments."""
    base: dict[str, Any] = {
        "model_id": model_id,
        "provider": provider,
        "extraction_passes": 1,
        "max_char_buffer": 2000,
    }
    if provider == "ollama":
        base["use_ollama"] = True
        base["model_url"] = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    else:
        base["use_ollama"] = False
        base["model_url"] = None
    return base


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def run_eval_programmatic(
    samples_dir: str | Path,
    output_dir: str | Path,
    model_id: str,
    provider: str,
    match_fn: str,
    skip_extraction: bool,
) -> dict[str, Any]:
    samples_dir = Path(samples_dir).resolve()
    output_dir = Path(output_dir).resolve()
    extract_kw = _extract_kwargs_for_provider(provider, model_id)

    per_file_reports: list[dict[str, Any]] = []
    all_pred: list[dict[str, Any]] = []
    all_gold: list[dict[str, Any]] = []

    txt_paths = sorted(samples_dir.glob("*.txt"))
    for txt_path in txt_paths:
        gold_path = txt_path.with_suffix(".json")
        if not gold_path.exists():
            continue

        gold = load_gold(gold_path)
        pred_path = txt_path.with_name(f"{txt_path.stem}_pred.json")

        if not skip_extraction:
            text = txt_path.read_text(encoding="utf-8", errors="replace")
            result = extract(text, **extract_kw)
            pred_list = extractions_to_serializable(result)
            _write_json(pred_path, pred_list)
        else:
            if not pred_path.exists():
                continue
            with open(pred_path, encoding="utf-8") as f:
                raw = json.load(f)
            pred_list = raw if isinstance(raw, list) else raw.get("extractions", [])

        metrics = compute_metrics(pred_list, gold, match_fn=match_fn)
        per_class = compute_per_class_metrics(pred_list, gold, match_fn=match_fn)
        per_file_reports.append({"file": txt_path.name, "metrics": metrics, "per_class": per_class})
        all_pred.extend(pred_list)
        all_gold.extend(gold)

    aggregate = compute_metrics(all_pred, all_gold, match_fn=match_fn) if all_pred or all_gold else {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "tp": 0,
        "pred": 0,
        "gold": 0,
    }
    per_class_aggregate = compute_per_class_metrics(all_pred, all_gold, match_fn=match_fn) if all_pred or all_gold else {}

    report: dict[str, Any] = {
        "config": {
            "samples": str(samples_dir),
            "model": model_id,
            "provider": provider,
            "match": match_fn,
            "skip_extraction": skip_extraction,
        },
        "per_file": per_file_reports,
        "aggregate": aggregate,
        "per_class_aggregate": per_class_aggregate,
    }
    _write_json(output_dir / "eval_results.json", report)
    return report


def _print_rich_tables(report: dict[str, Any], console: Console) -> None:
    per_file_reports = report.get("per_file", [])
    aggregate = report.get("aggregate", {})
    per_class_aggregate = report.get("per_class_aggregate", {})

    t_files = Table(title="Per-file aggregate P/R/F1")
    t_files.add_column("File", style="cyan")
    t_files.add_column("P", justify="right")
    t_files.add_column("R", justify="right")
    t_files.add_column("F1", justify="right")
    t_files.add_column("TP", justify="right")
    t_files.add_column("Pred", justify="right")
    t_files.add_column("Gold", justify="right")
    for row in per_file_reports:
        m = row.get("metrics", {})
        t_files.add_row(
            row.get("file", ""),
            f"{m.get('precision', 0.0):.4f}",
            f"{m.get('recall', 0.0):.4f}",
            f"{m.get('f1', 0.0):.4f}",
            str(m.get("tp", 0)),
            str(m.get("pred", 0)),
            str(m.get("gold", 0)),
        )
    console.print(t_files)

    t_agg = Table(title="Overall aggregate")
    t_agg.add_column("Metric", style="cyan")
    t_agg.add_column("Value", justify="right")
    for k in ("precision", "recall", "f1", "tp", "pred", "gold"):
        val = aggregate.get(k, 0.0 if k in ("precision", "recall", "f1") else 0)
        t_agg.add_row(k, f"{val:.4f}" if k in ("precision", "recall", "f1") else str(val))
    console.print(t_agg)

    t_pc = Table(title="Per-class breakdown (aggregate)")
    t_pc.add_column("Class", style="cyan")
    t_pc.add_column("P", justify="right")
    t_pc.add_column("R", justify="right")
    t_pc.add_column("F1", justify="right")
    t_pc.add_column("TP", justify="right")
    t_pc.add_column("Pred", justify="right")
    t_pc.add_column("Gold", justify="right")
    for cls in sorted(per_class_aggregate.keys()):
        m = per_class_aggregate[cls]
        t_pc.add_row(
            cls,
            f"{m.get('precision', 0.0):.4f}",
            f"{m.get('recall', 0.0):.4f}",
            f"{m.get('f1', 0.0):.4f}",
            str(m.get("tp", 0)),
            str(m.get("pred", 0)),
            str(m.get("gold", 0)),
        )
    console.print(t_pc)


def main() -> None:
    parser = argparse.ArgumentParser(description="ClinicalExtract evaluation on samples with gold JSON.")
    parser.add_argument("--samples", default="samples", help="Directory with .txt files and matching .json gold")
    parser.add_argument("--output", default="eval_results", help="Directory for eval_results.json")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Model id")
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=("gemini", "openai", "anthropic", "ollama"),
        help="Backend provider",
    )
    parser.add_argument("--match", default="partial", choices=("partial", "exact"), help="Text match mode")
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Do not call the extractor; only score existing *_pred.json files",
    )
    args = parser.parse_args()

    console = Console()
    report = run_eval_programmatic(
        samples_dir=args.samples,
        output_dir=args.output,
        model_id=args.model,
        provider=args.provider,
        match_fn=args.match,
        skip_extraction=args.skip_extraction,
    )
    console.print(f"[green]Wrote[/green] {Path(args.output).resolve() / 'eval_results.json'}")
    _print_rich_tables(report, console)


if __name__ == "__main__":
    main()
