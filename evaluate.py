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


def _extract_kwargs_for_provider(
    provider: str,
    model_id: str,
) -> dict[str, Any]:
    """Map CLI provider to extractor.extract() keyword arguments."""
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
    # Keys read from env inside extract() for gemini / openai / anthropic
    return base


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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

    samples_dir = Path(args.samples).resolve()
    output_dir = Path(args.output).resolve()
    match_fn = args.match

    console = Console()
    extract_kw = _extract_kwargs_for_provider(args.provider, args.model)

    per_file_reports: list[dict[str, Any]] = []
    all_pred: list[dict[str, Any]] = []
    all_gold: list[dict[str, Any]] = []

    txt_paths = sorted(samples_dir.glob("*.txt"))
    if not txt_paths:
        console.print(f"[yellow]No .txt files under {samples_dir}[/yellow]")
        return

    for txt_path in txt_paths:
        gold_path = txt_path.with_suffix(".json")
        if not gold_path.exists():
            continue

        gold = load_gold(gold_path)
        pred_path = txt_path.with_name(f"{txt_path.stem}_pred.json")

        if not args.skip_extraction:
            text = txt_path.read_text(encoding="utf-8", errors="replace")
            try:
                result = extract(text, **extract_kw)
                pred_list = extractions_to_serializable(result)
                _write_json(pred_path, pred_list)
            except Exception as e:
                console.print(f"[red]{txt_path.name}[/red] extraction failed: {e}")
                continue
        else:
            if not pred_path.exists():
                console.print(f"[yellow]Skip {txt_path.name}: no {pred_path.name} (--skip-extraction)[/yellow]")
                continue
            with open(pred_path, encoding="utf-8") as f:
                raw = json.load(f)
            pred_list = raw if isinstance(raw, list) else raw.get("extractions", [])

        metrics = compute_metrics(pred_list, gold, match_fn=match_fn)
        per_class = compute_per_class_metrics(pred_list, gold, match_fn=match_fn)

        per_file_reports.append(
            {
                "file": txt_path.name,
                "metrics": metrics,
                "per_class": per_class,
            }
        )
        all_pred.extend(pred_list)
        all_gold.extend(gold)

    if not per_file_reports:
        console.print("[yellow]No sample pairs (.txt + .json gold) evaluated.[/yellow]")
        return

    aggregate = compute_metrics(all_pred, all_gold, match_fn=match_fn)
    per_class_aggregate = compute_per_class_metrics(all_pred, all_gold, match_fn=match_fn)

    report: dict[str, Any] = {
        "config": {
            "samples": str(samples_dir),
            "model": args.model,
            "provider": args.provider,
            "match": match_fn,
            "skip_extraction": args.skip_extraction,
        },
        "per_file": per_file_reports,
        "aggregate": aggregate,
        "per_class_aggregate": per_class_aggregate,
    }

    out_json = output_dir / "eval_results.json"
    _write_json(out_json, report)
    console.print(f"[green]Wrote[/green] {out_json}")

    # --- Rich tables ---
    t_files = Table(title="Per-file aggregate P/R/F1")
    t_files.add_column("File", style="cyan")
    t_files.add_column("P", justify="right")
    t_files.add_column("R", justify="right")
    t_files.add_column("F1", justify="right")
    t_files.add_column("TP", justify="right")
    t_files.add_column("Pred", justify="right")
    t_files.add_column("Gold", justify="right")
    for row in per_file_reports:
        m = row["metrics"]
        t_files.add_row(
            row["file"],
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1']:.4f}",
            str(m["tp"]),
            str(m["pred"]),
            str(m["gold"]),
        )
    console.print(t_files)

    t_agg = Table(title="Overall aggregate")
    t_agg.add_column("Metric", style="cyan")
    t_agg.add_column("Value", justify="right")
    for k in ("precision", "recall", "f1", "tp", "pred", "gold"):
        t_agg.add_row(k, f"{aggregate[k]:.4f}" if k in ("precision", "recall", "f1") else str(aggregate[k]))
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
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1']:.4f}",
            str(m["tp"]),
            str(m["pred"]),
            str(m["gold"]),
        )
    console.print(t_pc)


if __name__ == "__main__":
    main()
