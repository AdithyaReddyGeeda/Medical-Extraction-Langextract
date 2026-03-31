from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

import evaluate


st.title("Evaluate")
st.caption("Run extraction + scoring against sample gold annotations directly in the browser.")

with st.sidebar:
    provider_label = st.selectbox(
        "Provider",
        ["Ollama (local)", "Gemini (cloud)", "OpenAI (cloud)", "Anthropic (cloud)"],
    )
    model_id = st.text_input("Model", value="gemini-2.5-flash")
    match_fn = st.selectbox("Match mode", ["partial", "exact"], index=0)
    skip_extraction = st.checkbox("Skip extraction (score existing _pred.json files only)", value=False)

provider_map = {
    "Ollama (local)": "ollama",
    "Gemini (cloud)": "gemini",
    "OpenAI (cloud)": "openai",
    "Anthropic (cloud)": "anthropic",
}
provider = provider_map[provider_label]

samples_dir = Path("samples").resolve()
output_dir = Path("eval_results").resolve()
st.write(f"Samples directory: `{samples_dir}`")

pairs = []
for txt_path in sorted(samples_dir.glob("*.txt")):
    gold_path = txt_path.with_suffix(".json")
    if gold_path.exists():
        pairs.append({"text_file": txt_path.name, "gold_file": gold_path.name})

if not pairs:
    st.info("No .txt + .json gold sample pairs found in samples/.")
else:
    st.dataframe(pd.DataFrame(pairs), use_container_width=True, hide_index=True)

run_clicked = st.button("Run Evaluation", type="primary", use_container_width=True, disabled=not pairs)
if run_clicked:
    with st.spinner("Running evaluation..."):
        report = evaluate.run_eval_programmatic(
            samples_dir=samples_dir,
            output_dir=output_dir,
            model_id=model_id,
            provider=provider,
            match_fn=match_fn,
            skip_extraction=skip_extraction,
        )

    per_file_rows = []
    for row in report.get("per_file", []):
        m = row.get("metrics", {})
        per_file_rows.append(
            {
                "file": row.get("file", ""),
                "precision": m.get("precision", 0.0),
                "recall": m.get("recall", 0.0),
                "f1": m.get("f1", 0.0),
                "tp": m.get("tp", 0),
                "pred": m.get("pred", 0),
                "gold": m.get("gold", 0),
            }
        )
    st.subheader("Per-file P/R/F1")
    st.dataframe(pd.DataFrame(per_file_rows), use_container_width=True, hide_index=True)

    per_class_rows = []
    for cls, m in sorted(report.get("per_class_aggregate", {}).items()):
        per_class_rows.append(
            {
                "class": cls,
                "precision": m.get("precision", 0.0),
                "recall": m.get("recall", 0.0),
                "f1": m.get("f1", 0.0),
                "tp": m.get("tp", 0),
                "pred": m.get("pred", 0),
                "gold": m.get("gold", 0),
            }
        )
    st.subheader("Per-class breakdown (aggregate)")
    st.dataframe(pd.DataFrame(per_class_rows), use_container_width=True, hide_index=True)

    agg = report.get("aggregate", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{agg.get('precision', 0.0):.4f}")
    c2.metric("Recall", f"{agg.get('recall', 0.0):.4f}")
    c3.metric("F1", f"{agg.get('f1', 0.0):.4f}")

    st.download_button(
        "Download eval JSON",
        json.dumps(report, indent=2),
        file_name="eval_results.json",
        mime="application/json",
    )
