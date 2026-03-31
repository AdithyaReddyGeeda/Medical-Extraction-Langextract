import pytest


@pytest.fixture
def sample_rows():
    """Minimal extraction rows for reuse across tests."""
    return [
        {"class": "medication", "text": "Lisinopril", "start": 0, "end": 10,
         "attributes": {"medication_group": "Lisinopril"}, "snippet": "Lisinopril"},
        {"class": "dosage", "text": "10 mg", "start": 11, "end": 16,
         "attributes": {"medication_group": "Lisinopril"}, "snippet": "10 mg"},
        {"class": "route", "text": "PO", "start": 17, "end": 19,
         "attributes": {"medication_group": "Lisinopril"}, "snippet": "PO"},
        {"class": "diagnosis", "text": "Hypertension", "start": 20, "end": 32,
         "attributes": {}, "snippet": "Hypertension"},
        {"class": "lab_test", "text": "WBC", "start": 33, "end": 36,
         "attributes": {"lab_group": "WBC"}, "snippet": "WBC"},
        {"class": "lab_value", "text": "12.2 K/uL", "start": 37, "end": 46,
         "attributes": {"lab_group": "WBC"}, "snippet": "12.2 K/uL"},
        {"class": "symptom_sign", "text": "fever", "start": 47, "end": 52,
         "attributes": {"negated": "true"}, "snippet": "fever"},
        {"class": "adverse_event_allergy", "text": "Penicillin - rash", "start": 53,
         "end": 70, "attributes": {}, "snippet": "Penicillin - rash"},
    ]


@pytest.fixture
def gold_rows():
    return [
        {"class": "medication", "text": "Lisinopril"},
        {"class": "dosage", "text": "10 mg"},
        {"class": "diagnosis", "text": "Hypertension"},
        {"class": "lab_test", "text": "WBC"},
    ]
