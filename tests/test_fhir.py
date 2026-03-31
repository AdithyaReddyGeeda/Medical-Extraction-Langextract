from utils.fhir import rows_to_fhir_bundle


def test_bundle_structure(sample_rows):
    bundle = rows_to_fhir_bundle(sample_rows)
    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert isinstance(bundle["entry"], list)
    assert len(bundle["entry"]) > 0


def test_all_entries_have_resource(sample_rows):
    bundle = rows_to_fhir_bundle(sample_rows)
    for entry in bundle["entry"]:
        assert "resource" in entry
        assert "resourceType" in entry["resource"]


def test_medication_statement_created(sample_rows):
    bundle = rows_to_fhir_bundle(sample_rows)
    types = [e["resource"]["resourceType"] for e in bundle["entry"]]
    assert "MedicationStatement" in types


def test_condition_created(sample_rows):
    bundle = rows_to_fhir_bundle(sample_rows)
    types = [e["resource"]["resourceType"] for e in bundle["entry"]]
    assert "Condition" in types


def test_observation_created(sample_rows):
    bundle = rows_to_fhir_bundle(sample_rows)
    types = [e["resource"]["resourceType"] for e in bundle["entry"]]
    assert "Observation" in types


def test_allergy_created(sample_rows):
    bundle = rows_to_fhir_bundle(sample_rows)
    types = [e["resource"]["resourceType"] for e in bundle["entry"]]
    assert "AllergyIntolerance" in types


def test_empty_rows_returns_empty_bundle():
    bundle = rows_to_fhir_bundle([])
    assert bundle["resourceType"] == "Bundle"
    assert bundle["entry"] == []


def test_patient_id_propagated(sample_rows):
    bundle = rows_to_fhir_bundle(sample_rows, patient_id="test-patient-42")
    for entry in bundle["entry"]:
        r = entry["resource"]
        subject = r.get("subject") or r.get("patient")
        if subject:
            assert "test-patient-42" in subject.get("reference", "")
