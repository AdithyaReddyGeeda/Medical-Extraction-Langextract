from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from utils.export import rows_to_excel_bytes


if "history" not in st.session_state:
    st.session_state.history = []

st.title("Extraction History")

history = st.session_state.history
if not history:
    st.info("No extractions yet. Go to Extract to run one.")
else:
    summary_rows = []
    total = len(history)
    for idx, item in enumerate(reversed(history), start=1):
        summary_rows.append(
            {
                "Run #": total - idx + 1,
                "Provider": item.get("provider", ""),
                "Model": item.get("model", ""),
                "Note preview": str(item.get("label", ""))[:50],
                "Entity count": item.get("count", 0),
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    for idx, item in enumerate(reversed(history), start=1):
        rows = item.get("rows", [])
        provider = item.get("provider", "")
        label = item.get("label", "")
        count = item.get("count", 0)
        run_num = total - idx + 1
        with st.expander(f"Run {run_num} · [{provider}] {label} · {count} entities"):
            st.caption(f"Model: {item.get('model', '')}")
            if rows:
                df = pd.DataFrame(rows)
                cols = [c for c in ("class", "text", "start", "end") if c in df.columns]
                st.dataframe(df[cols] if cols else df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download JSON",
                json.dumps(rows, indent=2),
                file_name=f"clinical_extractions_run_{run_num}.json",
                mime="application/json",
                key=f"hist_json_{run_num}",
            )
            st.download_button(
                "Download Excel",
                rows_to_excel_bytes(rows),
                file_name=f"clinical_extractions_run_{run_num}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"hist_xlsx_{run_num}",
            )

    if st.button("Clear all history", type="secondary"):
        st.session_state.history = []
        st.rerun()
