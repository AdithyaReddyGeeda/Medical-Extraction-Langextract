# -*- coding: utf-8 -*-
"""
ClinicalExtract — Medical Information Extraction with LangExtract

Streamlit app: upload clinical text, run extraction (Ollama or Gemini),
view structured output, highlighted spans, and export JSON/CSV.
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from extractor import extract, extractions_to_serializable, get_clinical_examples
from utils.visualization import (
    get_visualization_html_for_streamlit,
    save_annotated_documents_jsonl,
    generate_html_visualization,
)

st.set_page_config(
    page_title="ClinicalExtract",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ClinicalExtract — Medical Information Extraction with LangExtract")
st.caption("Turn clinical notes into structured JSON with precise source grounding. Powered by [LangExtract](https://github.com/google/langextract).")

# ---------------------------------------------------------------------------
# Sidebar: model and settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Model & settings")
    provider = st.selectbox(
        "Provider",
        ["Ollama (local)", "Gemini (cloud)"],
        help="Ollama requires a running local server (e.g. ollama serve).",
    )
    use_ollama = provider == "Ollama (local)"

    if use_ollama:
        model_id = st.selectbox(
            "Ollama model",
            [
                "qwen2.5-coder:32b-instruct",
                "llama3.1:70b",
                "gemma2:27b",
                "gemma2:2b",
                "llama3.2:3b",
            ],
            index=0,
        )
        model_url = st.text_input("Ollama URL", value="http://localhost:11434", help="Base URL of Ollama server.")
    else:
        model_id = st.selectbox(
            "Gemini model",
            ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
            index=0,
        )
        model_url = None

    st.divider()
    extraction_passes = st.slider("Extraction passes (long docs)", min_value=1, max_value=5, value=1)
    max_char_buffer = st.number_input("Max chunk size (chars)", min_value=500, max_value=10000, value=2000, step=500)

# ---------------------------------------------------------------------------
# Input: text area or file upload
# ---------------------------------------------------------------------------
input_source = st.radio("Input", ["Paste text", "Upload file", "Sample note"], horizontal=True)

text = ""
if input_source == "Paste text":
    text = st.text_area(
        "Clinical note",
        height=200,
        placeholder="Paste discharge summary, progress note, or radiology report here...",
    )
elif input_source == "Upload file":
    uploaded = st.file_uploader("Upload .txt file", type=["txt"])
    if uploaded:
        text = uploaded.read().decode("utf-8", errors="replace")
        st.text_area("Preview", text[:2000] + ("..." if len(text) > 2000 else ""), height=120, disabled=True)
else:
    samples_dir = Path(__file__).resolve().parent / "samples"
    sample_files = list(samples_dir.glob("*.txt"))
    if sample_files:
        sample_choice = st.selectbox(
            "Choose sample",
            sample_files,
            format_func=lambda p: p.name,
        )
        if sample_choice:
            text = sample_choice.read_text(encoding="utf-8", errors="replace")
        st.text_area("Sample content", text[:1500] + ("..." if len(text) > 1500 else ""), height=150, disabled=True)
    else:
        st.info("No .txt files in samples/. Add some or paste text.")
        text = ""

extract_clicked = st.button("Extract", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Run extraction and display results
# ---------------------------------------------------------------------------
if extract_clicked and text.strip():
    with st.spinner("Running extraction..."):
        try:
            result = extract(
                text,
                model_id=model_id,
                model_url=model_url if use_ollama else None,
                use_ollama=use_ollama,
                extraction_passes=extraction_passes,
                max_char_buffer=max_char_buffer,
            )
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            if use_ollama:
                st.info("Ensure Ollama is running (e.g. `ollama serve`) and the model is pulled (e.g. `ollama pull qwen2.5-coder:32b-instruct`).")
            else:
                st.info("Set LANGEXTRACT_API_KEY for Gemini (e.g. in .env or environment).")
            st.stop()

    rows = extractions_to_serializable(result)
    st.success(f"Extracted {len(rows)} entities.")

    # Tabs: Raw text | Structured (table + JSON) | Visualization | Evidence
    tab1, tab2, tab3, tab4 = st.tabs(["Raw text", "Structured output", "Visualization", "Evidence / Spans"])

    with tab1:
        st.text_area("Original text", result.text, height=300, disabled=True)

    with tab2:
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            st.subheader("JSON")
            st.json(rows)
            # Download JSON
            buf = io.StringIO()
            json.dump(rows, buf, indent=2)
            st.download_button("Download JSON", buf.getvalue(), file_name="clinical_extractions.json", mime="application/json")
            # Download CSV
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button("Download CSV", csv_buf.getvalue(), file_name="clinical_extractions.csv", mime="text/csv")
        else:
            st.info("No extractions returned.")

    with tab3:
        try:
            with tempfile.TemporaryDirectory(prefix="clinical_extract_") as tmp:
                jsonl_path = save_annotated_documents_jsonl(
                    [result],
                    output_name="extraction.jsonl",
                    output_dir=tmp,
                )
                html_content = generate_html_visualization(jsonl_path)
                st.components.v1.html(html_content, height=600, scrolling=True)
        except Exception as viz_err:
            st.warning(f"Visualization could not be generated: {viz_err}")

    with tab4:
        for r in rows:
            snippet = r.get("snippet", r.get("text", ""))
            start = r.get("start")
            end = r.get("end")
            st.markdown(f"**{r.get('class', '')}** (chars {start}-{end}): `{snippet}`")
            if r.get("attributes"):
                st.caption(f"Attributes: {r['attributes']}")

else:
    if extract_clicked and not text.strip():
        st.warning("Enter or upload clinical text first.")

# Footer
st.divider()
st.caption("ClinicalExtract uses LangExtract for grounded extraction. Not for clinical decision-making. Always verify with a qualified professional.")
