from extractor import (
    _normalize_attributes,
    _parse_json_from_model_text,
    rows_to_annotated_document,
)


def test_parse_json_plain():
    raw = '{"extractions": []}'
    result = _parse_json_from_model_text(raw)
    assert result == {"extractions": []}


def test_parse_json_with_fences():
    raw = '```json\n{"extractions": [{"extraction_class": "medication"}]}\n```'
    result = _parse_json_from_model_text(raw)
    assert "extractions" in result


def test_normalize_attributes_dict():
    result = _normalize_attributes({"medication_group": "Lisinopril"})
    assert result == {"medication_group": "Lisinopril"}


def test_normalize_attributes_none():
    assert _normalize_attributes(None) is None


def test_normalize_attributes_empty():
    assert _normalize_attributes({}) is None


def test_rows_to_annotated_document_roundtrip(sample_rows):
    text = "Lisinopril 10 mg PO Hypertension WBC 12.2 K/uL fever Penicillin - rash"
    doc = rows_to_annotated_document(sample_rows, text)
    assert doc.text == text
    assert len(doc.extractions) == len(sample_rows)
    assert doc.extractions[0].extraction_class == "medication"
    assert doc.extractions[0].extraction_text == "Lisinopril"
