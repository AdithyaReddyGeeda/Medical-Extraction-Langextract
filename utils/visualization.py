# -*- coding: utf-8 -*-
"""
Visualization helpers for ClinicalExtract.

Uses LangExtract's built-in HTML visualization and provides helpers
for Streamlit embedding (iframe or components).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import langextract as lx

if TYPE_CHECKING:
    from langextract.data import AnnotatedDocument


def save_annotated_documents_jsonl(
    results: list[AnnotatedDocument],
    output_name: str = "extraction_results.jsonl",
    output_dir: str | Path = ".",
) -> Path:
    """Save extraction results to a JSONL file (for LangExtract visualize)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / output_name
    lx.io.save_annotated_documents(results, output_name=output_name, output_dir=str(output_dir))
    return out_path


def generate_html_visualization(
    jsonl_path: str | Path,
) -> str:
    """
    Generate LangExtract's interactive HTML visualization from a JSONL file.

    Returns the HTML content as a string (for writing to file or embedding in Streamlit).
    """
    jsonl_path = str(jsonl_path)
    html_content = lx.visualize(jsonl_path)
    if hasattr(html_content, "data"):
        return html_content.data
    return html_content


def write_visualization_file(
    jsonl_path: str | Path,
    html_path: str | Path | None = None,
) -> Path:
    """Write the HTML visualization to a file. Returns path to the HTML file."""
    jsonl_path = Path(jsonl_path)
    if html_path is None:
        html_path = jsonl_path.with_suffix(".html")
    else:
        html_path = Path(html_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    content = generate_html_visualization(jsonl_path)
    html_path.write_text(content, encoding="utf-8")
    return html_path


def get_visualization_html_for_streamlit(
    results: list[AnnotatedDocument],
    temp_dir: str | Path | None = None,
) -> tuple[str, Path]:
    """
    Save results to temp JSONL, generate HTML, and return (html_string, jsonl_path).

    Useful for Streamlit: use st.components.v1.html(html_string, height=600)
    or write to a temp file and serve via iframe.
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="clinical_extract_")
    temp_dir = Path(temp_dir)
    jsonl_path = save_annotated_documents_jsonl(
        results,
        output_name="extraction.jsonl",
        output_dir=temp_dir,
    )
    html_content = generate_html_visualization(jsonl_path)
    return html_content, jsonl_path
