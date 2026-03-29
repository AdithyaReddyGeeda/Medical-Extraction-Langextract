# -*- coding: utf-8 -*-
"""
ClinicalExtract — Medical Information Extraction with LangExtract

Streamlit app: upload clinical text, run extraction (Ollama or Gemini),
view structured output, highlighted spans, and export JSON/CSV.
"""
from __future__ import annotations

import io
import os

from dotenv import load_dotenv
load_dotenv()
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from extractor import extract, extractions_to_serializable, rows_to_annotated_document
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

if "history" not in st.session_state:
    st.session_state.history = []

st.title("ClinicalExtract — Medical Information Extraction with LangExtract")
st.caption("Turn clinical notes into structured JSON with precise source grounding. Powered by [LangExtract](https://github.com/google/langextract).")


def parse_document_bytes(filename: str, raw: bytes) -> str:
    """Decode uploaded bytes to plain text (txt / pdf / docx). Raises on unsupported type or parse failure."""
    name = filename.lower()
    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="replace")
    if name.endswith(".pdf"):
        import pdfplumber

        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            return "\n".join((p.extract_text() or "") for p in pdf.pages)
    if name.endswith(".docx"):
        import docx

        doc = docx.Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs)
    raise ValueError("Unsupported file type; use .txt, .pdf, or .docx.")


@st.cache_data(show_spinner=False)
def _cached_extract(
    text: str,
    model_id: str,
    model_url: str | None,
    use_ollama: bool,
    extraction_passes: int,
    max_char_buffer: int,
    provider: str,
    openai_api_key: str | None,
    anthropic_api_key: str | None,
) -> tuple[list[dict], str]:
    """Cache extraction by inputs; returns serializable rows and document text."""
    result = extract(
        text,
        model_id=model_id,
        provider=provider,
        model_url=model_url if use_ollama else None,
        use_ollama=use_ollama,
        extraction_passes=extraction_passes,
        max_char_buffer=max_char_buffer,
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
    )
    return extractions_to_serializable(result), result.text or ""


def _group_key(row: dict) -> str | None:
    attrs = row.get("attributes") or {}
    return attrs.get("medication_group") or attrs.get("lab_group")


def _render_evidence_group(label: str, group_rows: list[dict]) -> None:
    with st.expander(label=label):
        if group_rows:
            summary = [
                {
                    "class": r.get("class", ""),
                    "text": r.get("text", ""),
                    "start": r.get("start"),
                    "end": r.get("end"),
                }
                for r in group_rows
            ]
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
        for r in group_rows:
            snippet = r.get("snippet", r.get("text", ""))
            start = r.get("start")
            end = r.get("end")
            st.markdown(f"**{r.get('class', '')}** (chars {start}-{end}): `{snippet}`")
            if r.get("attributes"):
                st.caption(f"Attributes: {r['attributes']}")


def _history_label(note_text: str) -> str:
    t = note_text.strip()
    return (t[:60] + "...") if len(t) > 60 else t


# ---------------------------------------------------------------------------
# Sidebar: model and settings
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Model & settings")
    provider_label = st.selectbox(
        "Provider",
        ["Ollama (local)", "Gemini (cloud)", "OpenAI (cloud)", "Anthropic (cloud)"],
        help="Ollama requires a running local server (e.g. ollama serve).",
    )
    use_ollama = provider_label == "Ollama (local)"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    if use_ollama:
        provider = "ollama"
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
    elif provider_label == "Gemini (cloud)":
        provider = "gemini"
        model_id = st.selectbox(
            "Gemini model",
            ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
            index=0,
        )
        model_url = None
    elif provider_label == "OpenAI (cloud)":
        provider = "openai"
        model_id = st.selectbox("OpenAI model", ["gpt-4o", "gpt-4o-mini"], index=0)
        model_url = None
        openai_api_key = st.text_input(
            "OPENAI_API_KEY",
            value=os.environ.get("OPENAI_API_KEY", ""),
            help="OpenAI API key (or set OPENAI_API_KEY in the environment).",
        )
        if openai_api_key == "":
            openai_api_key = None
    else:
        provider = "anthropic"
        model_id = st.selectbox(
            "Anthropic model",
            ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"],
            index=0,
        )
        model_url = None
        anthropic_api_key = st.text_input(
            "ANTHROPIC_API_KEY",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            help="Anthropic API key (or set ANTHROPIC_API_KEY in the environment).",
        )
        if anthropic_api_key == "":
            anthropic_api_key = None

    st.divider()
    extraction_passes = st.slider("Extraction passes (long docs)", min_value=1, max_value=5, value=1)
    max_char_buffer = st.number_input("Max chunk size (chars)", min_value=500, max_value=10000, value=2000, step=500)

    st.divider()
    st.subheader("History")
    if not st.session_state.history:
        st.caption("No extractions yet.")
    else:
        for hi, item in enumerate(reversed(st.session_state.history)):
            exp_label = f"[{item['provider']}] {item['label']} ({item['count']} entities)"
            with st.expander(exp_label):
                st.caption(f"Model: {item['model']}")
                mini = item.get("rows") or []
                if mini:
                    df_mini = pd.DataFrame(mini)
                    cols = [c for c in ("class", "text") if c in df_mini.columns]
                    if cols:
                        st.dataframe(df_mini[cols], use_container_width=True, hide_index=True)
                st.download_button(
                    "Re-download JSON",
                    json.dumps(item.get("rows", []), indent=2),
                    file_name="clinical_extractions.json",
                    mime="application/json",
                    key=f"hist_dl_{hi}",
                )
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Input: text area or file upload
# ---------------------------------------------------------------------------
input_source = st.radio(
    "Input",
    ["Paste text", "Upload file", "Sample note", "Batch upload"],
    horizontal=True,
)

text = ""
batch_files = None
extract_all_clicked = False

if input_source == "Paste text":
    text = st.text_area(
        "Clinical note",
        height=200,
        placeholder="Paste discharge summary, progress note, or radiology report here...",
    )
elif input_source == "Upload file":
    uploaded = st.file_uploader("Upload file", type=["txt", "pdf", "docx"])
    if uploaded:
        raw = uploaded.getvalue()
        try:
            text = parse_document_bytes(uploaded.name, raw)
        except Exception as parse_err:
            st.error(f"Could not read this file: {parse_err}. Try another file or paste text instead.")
        if text:
            st.text_area("Preview", text[:2000] + ("..." if len(text) > 2000 else ""), height=120, disabled=True)
elif input_source == "Batch upload":
    batch_files = st.file_uploader(
        "Upload files (batch)",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
    )
    extract_all_clicked = st.button("Extract All", type="primary", use_container_width=True)
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

extract_clicked = False
if input_source != "Batch upload":
    extract_clicked = st.button("Extract", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Batch extraction (separate flow; does not use session history)
# ---------------------------------------------------------------------------
if extract_all_clicked and input_source == "Batch upload":
    if not batch_files:
        st.warning("Upload at least one file before running batch extraction.")
    else:
        combined: list[dict] = []
        n = len(batch_files)
        progress_bar = st.progress(0)
        for bi, uploaded in enumerate(batch_files):
            fname = uploaded.name
            try:
                doc_text = parse_document_bytes(fname, uploaded.getvalue())
            except Exception as e:
                st.warning(f"{fname}: {e}")
                progress_bar.progress((bi + 1) / n)
                continue
            if not doc_text.strip():
                st.warning(f"{fname}: empty document after parsing.")
                progress_bar.progress((bi + 1) / n)
                continue
            try:
                rows, _rt = _cached_extract(
                    doc_text.strip(),
                    model_id,
                    model_url,
                    use_ollama,
                    extraction_passes,
                    max_char_buffer,
                    provider,
                    openai_api_key,
                    anthropic_api_key,
                )
                for r in rows:
                    row_copy = dict(r)
                    row_copy["file"] = fname
                    combined.append(row_copy)
            except Exception as e:
                st.warning(f"{fname}: {e}")
            progress_bar.progress((bi + 1) / n)
        progress_bar.empty()
        if combined:
            df_all = pd.DataFrame(combined)
            col_order = ["file"] + [c for c in df_all.columns if c != "file"]
            df_all = df_all[col_order]
            st.subheader("Batch results")
            st.dataframe(df_all, use_container_width=True)
            json_buf = io.StringIO()
            json.dump(combined, json_buf, indent=2)
            st.download_button(
                "Download combined JSON",
                json_buf.getvalue(),
                file_name="clinical_extractions_batch.json",
                mime="application/json",
            )
            csv_buf = io.StringIO()
            df_all.to_csv(csv_buf, index=False)
            st.download_button(
                "Download combined CSV",
                csv_buf.getvalue(),
                file_name="clinical_extractions_batch.csv",
                mime="text/csv",
            )

# ---------------------------------------------------------------------------
# Run extraction and display results (single-note modes)
# ---------------------------------------------------------------------------
if extract_clicked and text.strip():
    with st.spinner("Running extraction..."):
        try:
            rows, result_text = _cached_extract(
                text.strip(),
                model_id,
                model_url,
                use_ollama,
                extraction_passes,
                max_char_buffer,
                provider,
                openai_api_key,
                anthropic_api_key,
            )
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            if use_ollama:
                st.info("Ensure Ollama is running (e.g. `ollama serve`) and the model is pulled (e.g. `ollama pull qwen2.5-coder:32b-instruct`).")
            elif provider_label == "Gemini (cloud)":
                st.info("Set LANGEXTRACT_API_KEY for Gemini (e.g. in .env or environment).")
            elif provider_label == "OpenAI (cloud)":
                st.info("Set OPENAI_API_KEY (sidebar or environment).")
            elif provider_label == "Anthropic (cloud)":
                st.info("Set ANTHROPIC_API_KEY (sidebar or environment).")
            st.stop()

    st.session_state.history.append(
        {
            "label": _history_label(text),
            "model": model_id,
            "provider": provider,
            "count": len(rows),
            "rows": [dict(r) for r in rows],
            "text": result_text,
        }
    )

    result = rows_to_annotated_document(rows, result_text)
    st.success(f"Extracted {len(rows)} entities.")

    # Tabs: Raw text | Structured (table + JSON) | Visualization | Evidence
    tab1, tab2, tab3, tab4 = st.tabs(["Raw text", "Structured output", "Visualization", "Evidence / Spans"])

    with tab1:
        st.text_area("Original text", result.text or "", height=300, disabled=True)

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
        grouped: dict[str, list[dict]] = {}
        other: list[dict] = []
        for r in rows:
            g = _group_key(r)
            if g:
                grouped.setdefault(str(g), []).append(r)
            else:
                other.append(r)
        for gname in sorted(grouped.keys()):
            _render_evidence_group(gname, grouped[gname])
        if other:
            _render_evidence_group("Other", other)

else:
    if extract_clicked and not text.strip():
        st.warning("Enter or upload clinical text first.")

# Footer
st.divider()
st.caption("ClinicalExtract uses LangExtract for grounded extraction. Not for clinical decision-making. Always verify with a qualified professional.")
