# -*- coding: utf-8 -*-
"""
ClinicalExtract — Core extraction logic using LangExtract.

Defines clinical schema (medications, diagnoses, procedures, labs, etc.),
few-shot examples, and runs extraction with grounding and optional chunking.
"""
from __future__ import annotations

import json
import os
import re
import textwrap
from pathlib import Path
from typing import Any

import langextract as lx

# ---------------------------------------------------------------------------
# Clinical extraction schema (conceptual — enforced via few-shot examples)
# ---------------------------------------------------------------------------
# Entities we extract:
#   medications: name, strength/dosage, route, frequency, duration, reason, status
#   diagnoses: name, icd_suggestion, onset, status
#   procedures: name, date, laterality, findings
#   labs: test_name, value, unit, reference_range, interpretation
#   symptoms_signs: description
#   adverse_events_allergies: substance, reaction, severity
#   demographics: age, sex, dob (if present)
#   relationships: e.g. medication_group to link med → indication
# ---------------------------------------------------------------------------

CLINICAL_PROMPT = textwrap.dedent("""
Extract clinical entities from the text in order of appearance.
Use exact verbatim text for extractions when possible; do not paraphrase or overlap spans.

Entity types to extract:
- medication: drug name
- dosage: strength/amount (e.g. 10mg, 250 mg)
- route: route of administration (PO, IV, topical, etc.)
- frequency: how often (daily, BID, q4h, etc.)
- duration: length of use (e.g. for one week)
- indication: reason or condition the medication is for
- medication_status: active, discontinued, or as needed

- diagnosis: condition or diagnosis name
- diagnosis_icd: suggested ICD code if inferable (e.g. I10 for hypertension)
- diagnosis_status: active, resolved, history of
- diagnosis_onset: when it started if mentioned

- procedure: procedure or intervention name
- procedure_date: when performed if stated
- procedure_laterality: left, right, bilateral if applicable
- procedure_findings: key findings

- lab_test: name of test
- lab_value: numeric or categorical value
- lab_unit: unit (mg/dL, mmol/L, etc.)
- lab_reference: reference range if given
- lab_interpretation: high, low, normal, abnormal

- symptom_sign: symptom or sign description
- adverse_event_allergy: substance and reaction (e.g. Penicillin - rash)
- allergy_severity: mild, moderate, severe if stated

- demographic_age: age if mentioned
- demographic_sex: sex if mentioned
- demographic_dob: date of birth if mentioned

Use attributes to link related extractions (e.g. medication_group for all fields of one medication).
List entities in order of appearance. Do not invent text that is not in the source.
""").strip()

# Instructions appended for Anthropic JSON-only output (offsets aligned to source text).
_ANTHROPIC_JSON_SUFFIX = """
Return ONLY a single JSON object (no markdown code fences) with this exact shape:
{"extractions": [{"extraction_class": "string", "extraction_text": "string", "attributes": {}, "start_pos": 0, "end_pos": 0}]}
Use verbatim substrings from the document. start_pos and end_pos are 0-based character indices into the document text (end exclusive).
If an attribute value is not applicable, use an empty object for attributes.
""".strip()


def _example_discharge() -> lx.data.ExampleData:
    """Few-shot example: discharge summary snippet."""
    text = (
        "DISCHARGE SUMMARY\n"
        "Patient is a 67 yo M with HTN and DM2. "
        "He was given 250 mg IV Cefazolin TID for one week for cellulitis. "
        "Home medications: Lisinopril 10 mg PO daily for hypertension, Metformin 500 mg PO BID for diabetes. "
        "Allergies: Penicillin - rash. "
        "Discharge diagnosis: Cellulitis left lower leg, Hypertension, Type 2 diabetes."
    )
    return lx.data.ExampleData(
        text=text,
        extractions=[
            lx.data.Extraction(extraction_class="demographic_age", extraction_text="67 yo", attributes={}),
            lx.data.Extraction(extraction_class="demographic_sex", extraction_text="M", attributes={}),
            lx.data.Extraction(extraction_class="medication", extraction_text="Cefazolin", attributes={"medication_group": "Cefazolin"}),
            lx.data.Extraction(extraction_class="dosage", extraction_text="250 mg", attributes={"medication_group": "Cefazolin"}),
            lx.data.Extraction(extraction_class="route", extraction_text="IV", attributes={"medication_group": "Cefazolin"}),
            lx.data.Extraction(extraction_class="frequency", extraction_text="TID", attributes={"medication_group": "Cefazolin"}),
            lx.data.Extraction(extraction_class="duration", extraction_text="for one week", attributes={"medication_group": "Cefazolin"}),
            lx.data.Extraction(extraction_class="indication", extraction_text="cellulitis", attributes={"medication_group": "Cefazolin"}),
            lx.data.Extraction(extraction_class="medication", extraction_text="Lisinopril", attributes={"medication_group": "Lisinopril"}),
            lx.data.Extraction(extraction_class="dosage", extraction_text="10 mg", attributes={"medication_group": "Lisinopril"}),
            lx.data.Extraction(extraction_class="route", extraction_text="PO", attributes={"medication_group": "Lisinopril"}),
            lx.data.Extraction(extraction_class="frequency", extraction_text="daily", attributes={"medication_group": "Lisinopril"}),
            lx.data.Extraction(extraction_class="indication", extraction_text="hypertension", attributes={"medication_group": "Lisinopril"}),
            lx.data.Extraction(extraction_class="medication", extraction_text="Metformin", attributes={"medication_group": "Metformin"}),
            lx.data.Extraction(extraction_class="dosage", extraction_text="500 mg", attributes={"medication_group": "Metformin"}),
            lx.data.Extraction(extraction_class="route", extraction_text="PO", attributes={"medication_group": "Metformin"}),
            lx.data.Extraction(extraction_class="frequency", extraction_text="BID", attributes={"medication_group": "Metformin"}),
            lx.data.Extraction(extraction_class="indication", extraction_text="diabetes", attributes={"medication_group": "Metformin"}),
            lx.data.Extraction(extraction_class="adverse_event_allergy", extraction_text="Penicillin - rash", attributes={}),
            lx.data.Extraction(extraction_class="diagnosis", extraction_text="Cellulitis left lower leg", attributes={}),
            lx.data.Extraction(extraction_class="diagnosis", extraction_text="Hypertension", attributes={}),
            lx.data.Extraction(extraction_class="diagnosis", extraction_text="Type 2 diabetes", attributes={}),
        ],
    )


def _example_labs() -> lx.data.ExampleData:
    """Few-shot example: labs and vitals."""
    text = (
        "Labs: WBC 12.2 K/uL (ref 4.5-11), Hgb 10.1 g/dL (ref 12-16), low. "
        "Creatinine 1.4 mg/dL (ref 0.7-1.3), elevated. "
        "Rapid flu: negative."
    )
    return lx.data.ExampleData(
        text=text,
        extractions=[
            lx.data.Extraction(extraction_class="lab_test", extraction_text="WBC", attributes={"lab_group": "WBC"}),
            lx.data.Extraction(extraction_class="lab_value", extraction_text="12.2 K/uL", attributes={"lab_group": "WBC"}),
            lx.data.Extraction(extraction_class="lab_reference", extraction_text="ref 4.5-11", attributes={"lab_group": "WBC"}),
            lx.data.Extraction(extraction_class="lab_test", extraction_text="Hgb", attributes={"lab_group": "Hgb"}),
            lx.data.Extraction(extraction_class="lab_value", extraction_text="10.1 g/dL", attributes={"lab_group": "Hgb"}),
            lx.data.Extraction(extraction_class="lab_reference", extraction_text="ref 12-16", attributes={"lab_group": "Hgb"}),
            lx.data.Extraction(extraction_class="lab_interpretation", extraction_text="low", attributes={"lab_group": "Hgb"}),
            lx.data.Extraction(extraction_class="lab_test", extraction_text="Creatinine", attributes={"lab_group": "Creatinine"}),
            lx.data.Extraction(extraction_class="lab_value", extraction_text="1.4 mg/dL", attributes={"lab_group": "Creatinine"}),
            lx.data.Extraction(extraction_class="lab_reference", extraction_text="ref 0.7-1.3", attributes={"lab_group": "Creatinine"}),
            lx.data.Extraction(extraction_class="lab_interpretation", extraction_text="elevated", attributes={"lab_group": "Creatinine"}),
            lx.data.Extraction(extraction_class="lab_test", extraction_text="Rapid flu", attributes={}),
            lx.data.Extraction(extraction_class="lab_value", extraction_text="negative", attributes={}),
        ],
    )


def _example_procedures() -> lx.data.ExampleData:
    """Few-shot example: procedure note."""
    text = (
        "CT chest with contrast performed 01/15/2025. "
        "Findings: 2 cm nodule in the right upper lobe, no left-sided abnormalities. "
        "Echo: EF 55%, mild MR."
    )
    return lx.data.ExampleData(
        text=text,
        extractions=[
            lx.data.Extraction(extraction_class="procedure", extraction_text="CT chest with contrast", attributes={}),
            lx.data.Extraction(extraction_class="procedure_date", extraction_text="01/15/2025", attributes={}),
            lx.data.Extraction(extraction_class="procedure_findings", extraction_text="2 cm nodule in the right upper lobe", attributes={"laterality": "right"}),
            lx.data.Extraction(extraction_class="procedure_findings", extraction_text="no left-sided abnormalities", attributes={"laterality": "left"}),
            lx.data.Extraction(extraction_class="procedure", extraction_text="Echo", attributes={}),
            lx.data.Extraction(extraction_class="procedure_findings", extraction_text="EF 55%", attributes={}),
            lx.data.Extraction(extraction_class="procedure_findings", extraction_text="mild MR", attributes={}),
        ],
    )


def _example_symptoms() -> lx.data.ExampleData:
    """Few-shot example: symptoms and HPI snippet."""
    text = (
        "Patient presents with 3 days of fever, cough, and shortness of breath. "
        "Denies chest pain. No known drug allergies."
    )
    return lx.data.ExampleData(
        text=text,
        extractions=[
            lx.data.Extraction(extraction_class="symptom_sign", extraction_text="fever", attributes={}),
            lx.data.Extraction(extraction_class="symptom_sign", extraction_text="cough", attributes={}),
            lx.data.Extraction(extraction_class="symptom_sign", extraction_text="shortness of breath", attributes={}),
            lx.data.Extraction(extraction_class="symptom_sign", extraction_text="Denies chest pain", attributes={"negated": "true"}),
        ],
    )


def _example_radiology() -> lx.data.ExampleData:
    """Few-shot example: radiology report snippet."""
    text = (
        "CHEST X-RAY: PA and lateral views. "
        "Heart size normal. Lungs clear bilaterally. "
        "No pleural effusion or pneumothorax. "
        "Impression: No acute cardiopulmonary process."
    )
    return lx.data.ExampleData(
        text=text,
        extractions=[
            lx.data.Extraction(extraction_class="procedure", extraction_text="CHEST X-RAY", attributes={}),
            lx.data.Extraction(extraction_class="procedure_findings", extraction_text="Heart size normal", attributes={}),
            lx.data.Extraction(extraction_class="procedure_findings", extraction_text="Lungs clear bilaterally", attributes={}),
            lx.data.Extraction(extraction_class="procedure_findings", extraction_text="No pleural effusion or pneumothorax", attributes={}),
            lx.data.Extraction(extraction_class="diagnosis", extraction_text="No acute cardiopulmonary process", attributes={}),
        ],
    )


def _example_medication_only() -> lx.data.ExampleData:
    """Simple medication-only example (LangExtract-style)."""
    text = "Patient was given 250 mg IV Cefazolin TID for one week."
    return lx.data.ExampleData(
        text=text,
        extractions=[
            lx.data.Extraction(extraction_class="dosage", extraction_text="250 mg", attributes={"medication_group": "Cefazolin"}),
            lx.data.Extraction(extraction_class="route", extraction_text="IV", attributes={"medication_group": "Cefazolin"}),
            lx.data.Extraction(extraction_class="medication", extraction_text="Cefazolin", attributes={"medication_group": "Cefazolin"}),
            lx.data.Extraction(extraction_class="frequency", extraction_text="TID", attributes={"medication_group": "Cefazolin"}),
            lx.data.Extraction(extraction_class="duration", extraction_text="for one week", attributes={"medication_group": "Cefazolin"}),
        ],
    )


def get_clinical_examples() -> list[lx.data.ExampleData]:
    """Return 4–6 high-quality few-shot examples for clinical extraction."""
    return [
        _example_medication_only(),
        _example_discharge(),
        _example_labs(),
        _example_procedures(),
        _example_symptoms(),
        _example_radiology(),
    ]


def _serialize_examples_for_anthropic(examples: list[lx.data.ExampleData]) -> str:
    """Serialize few-shot examples as JSON for the Anthropic prompt."""
    blocks: list[dict[str, Any]] = []
    for ex in examples:
        ex_rows: list[dict[str, Any]] = []
        for e in ex.extractions:
            ex_rows.append(
                {
                    "extraction_class": e.extraction_class,
                    "extraction_text": e.extraction_text,
                    "attributes": dict(e.attributes) if e.attributes else {},
                }
            )
        blocks.append({"text": ex.text, "extractions": ex_rows})
    return json.dumps(blocks, indent=2, ensure_ascii=False)


def _parse_json_from_model_text(raw: str) -> dict[str, Any]:
    """Extract JSON object from model output (handles optional ```json fences)."""
    text = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def _normalize_attributes(attrs: Any) -> dict[str, str | list[str]] | None:
    """Ensure attributes match Extraction typing (str or list[str] values)."""
    if not attrs or not isinstance(attrs, dict):
        return None
    out: dict[str, str | list[str]] = {}
    for k, v in attrs.items():
        if isinstance(v, list):
            out[str(k)] = [str(x) for x in v]
        else:
            out[str(k)] = str(v)
    return out or None


def _align_char_interval(source: str, extraction_text: str, start_pos: Any, end_pos: Any) -> lx.data.CharInterval | None:
    """Prefer model offsets when valid; otherwise locate extraction_text in source."""
    if (
        isinstance(start_pos, int)
        and isinstance(end_pos, int)
        and 0 <= start_pos < end_pos <= len(source)
        and source[start_pos:end_pos] == extraction_text
    ):
        return lx.data.CharInterval(start_pos=start_pos, end_pos=end_pos)
    idx = source.find(extraction_text)
    if idx != -1:
        return lx.data.CharInterval(start_pos=idx, end_pos=idx + len(extraction_text))
    return None


def _extract_anthropic(
    text: str,
    model_id: str,
    api_key: str | None,
    *,
    max_tokens: int = 8192,
) -> lx.data.AnnotatedDocument:
    """Run extraction via Anthropic Messages API; returns AnnotatedDocument aligned to source text."""
    import anthropic

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Anthropic API key missing: set ANTHROPIC_API_KEY or pass anthropic_api_key.")

    examples = get_clinical_examples()
    few_shot_json = _serialize_examples_for_anthropic(examples)
    user_body = (
        f"{CLINICAL_PROMPT}\n\n"
        f"Few-shot examples (JSON):\n{few_shot_json}\n\n"
        f"Document to extract from:\n---\n{text}\n---\n\n"
        f"{_ANTHROPIC_JSON_SUFFIX}"
    )

    client = anthropic.Anthropic(api_key=key)
    message = client.messages.create(
        model=model_id,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": user_body}],
    )
    raw_text = ""
    for block in message.content:
        if block.type == "text":
            raw_text += block.text

    payload = _parse_json_from_model_text(raw_text)
    raw_extractions = payload.get("extractions")
    if not isinstance(raw_extractions, list):
        raise ValueError("Anthropic response JSON missing 'extractions' array.")

    extractions: list[lx.data.Extraction] = []
    for item in raw_extractions:
        if not isinstance(item, dict):
            continue
        ec = str(item.get("extraction_class", ""))
        et = str(item.get("extraction_text", ""))
        attrs = _normalize_attributes(item.get("attributes"))
        ci = _align_char_interval(text, et, item.get("start_pos"), item.get("end_pos"))
        extractions.append(
            lx.data.Extraction(
                extraction_class=ec,
                extraction_text=et,
                attributes=attrs,
                char_interval=ci,
            )
        )
    return lx.data.AnnotatedDocument(text=text, extractions=extractions)


def extract(
    text_or_path: str | Path,
    model_id: str = "gemini-2.5-flash",
    *,
    provider: str = "gemini",
    model_url: str | None = None,
    api_key: str | None = None,
    openai_api_key: str | None = None,
    anthropic_api_key: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 8192,
    extraction_passes: int = 1,
    max_workers: int = 4,
    max_char_buffer: int = 2000,
    use_ollama: bool = False,
) -> lx.data.AnnotatedDocument:
    """
    Run clinical extraction on raw text or a file path.

    Args:
        text_or_path: Clinical text string or path to .txt file.
        model_id: LLM model (e.g. gemini-2.5-flash, gpt-4o, or Ollama model id).
        provider: One of gemini, ollama, openai, anthropic.
        model_url: For Ollama, base URL (default http://localhost:11434).
        api_key: Optional API key for Gemini (else LANGEXTRACT_API_KEY).
        openai_api_key: API key for OpenAI (else OPENAI_API_KEY).
        anthropic_api_key: API key for Anthropic (else ANTHROPIC_API_KEY).
        temperature: LLM temperature.
        max_tokens: Max output tokens (Anthropic path).
        extraction_passes: Number of passes for long docs (improves recall).
        max_workers: Parallel workers for chunked extraction.
        max_char_buffer: Chunk size for long documents.
        use_ollama: If True, use Ollama (fence_output and use_schema_constraints False).

    Returns:
        AnnotatedDocument with .extractions and .text (and char_interval on each extraction).
    """
    if isinstance(text_or_path, Path):
        text_or_path = str(text_or_path)
    if os.path.isfile(text_or_path):
        with open(text_or_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        text = text_or_path

    if provider == "anthropic":
        return _extract_anthropic(
            text,
            model_id,
            anthropic_api_key,
            max_tokens=max_tokens,
        )

    examples = get_clinical_examples()

    if provider == "openai":
        oa_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        kwargs: dict[str, Any] = {
            "text_or_documents": text,
            "prompt_description": CLINICAL_PROMPT,
            "examples": examples,
            "model_id": model_id,
            "extraction_passes": extraction_passes,
            "max_workers": max_workers,
            "max_char_buffer": max_char_buffer,
            "fence_output": True,
            "use_schema_constraints": False,
            "api_key": oa_key,
        }
        result = lx.extract(**kwargs)
        return result

    # Gemini and Ollama use LangExtract with provider resolved from model_id / URL
    api_key = api_key or os.environ.get("LANGEXTRACT_API_KEY")

    kwargs = {
        "text_or_documents": text,
        "prompt_description": CLINICAL_PROMPT,
        "examples": examples,
        "model_id": model_id,
        "extraction_passes": extraction_passes,
        "max_workers": max_workers,
        "max_char_buffer": max_char_buffer,
    }
    if api_key:
        kwargs["api_key"] = api_key

    if use_ollama or provider == "ollama":
        kwargs["fence_output"] = False
        kwargs["use_schema_constraints"] = False
        if model_url:
            kwargs["model_url"] = model_url
    else:
        # Cloud Gemini: keep LangExtract defaults for schema / fences
        pass

    result = lx.extract(**kwargs)
    return result


def extractions_to_serializable(result: lx.data.AnnotatedDocument) -> list[dict[str, Any]]:
    """Turn AnnotatedDocument extractions into a list of dicts (for JSON/table export)."""
    out = []
    src = result.text or ""
    for e in result.extractions or []:
        item = {
            "class": e.extraction_class,
            "text": e.extraction_text,
            "attributes": dict(e.attributes) if e.attributes else {},
        }
        if e.char_interval is not None:
            item["start"] = e.char_interval.start_pos
            item["end"] = e.char_interval.end_pos
            item["snippet"] = src[e.char_interval.start_pos : e.char_interval.end_pos]
        out.append(item)
    return out


def rows_to_annotated_document(rows: list[dict[str, Any]], text: str) -> lx.data.AnnotatedDocument:
    """Rebuild AnnotatedDocument from cached serializable rows + source text (for visualization)."""
    extractions: list[lx.data.Extraction] = []
    for r in rows:
        attrs = r.get("attributes")
        if attrs is not None and not isinstance(attrs, dict):
            attrs = dict(attrs)
        ci = None
        if r.get("start") is not None and r.get("end") is not None:
            ci = lx.data.CharInterval(start_pos=int(r["start"]), end_pos=int(r["end"]))
        extractions.append(
            lx.data.Extraction(
                extraction_class=str(r.get("class", "")),
                extraction_text=str(r.get("text", "")),
                attributes=attrs if attrs else None,
                char_interval=ci,
            )
        )
    return lx.data.AnnotatedDocument(text=text, extractions=extractions)
