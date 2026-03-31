"""
Map ClinicalExtract rows to FHIR R4 resources and bundle them.

Supported mappings:
  medication / dosage / route / frequency / duration / indication
    -> MedicationStatement (grouped by medication_group attribute)
  diagnosis / diagnosis_status / diagnosis_onset / diagnosis_icd
    -> Condition (one per diagnosis extraction)
  lab_test / lab_value / lab_unit / lab_reference / lab_interpretation
    -> Observation (grouped by lab_group attribute)
  procedure / procedure_date / procedure_findings
    -> Procedure (grouped by position in rows)
  symptom_sign
    -> Condition (clinicalStatus: "active", category: "problem-list-item")
  adverse_event_allergy
    -> AllergyIntolerance
  demographic_age / demographic_sex
    -> Patient (single resource, merge all demographics)
"""
from __future__ import annotations

import re
import uuid
from typing import Any


def _slug(text: str) -> str:
    """Convert arbitrary text to an ID-safe token."""
    token = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return token or str(uuid.uuid4())[:8]


def _find_nearby(rows: list[dict[str, Any]], start_idx: int, klass: str, window: int) -> list[dict[str, Any]]:
    """Return rows of a class in [start_idx + 1, start_idx + window]."""
    out: list[dict[str, Any]] = []
    end = min(len(rows), start_idx + window + 1)
    for i in range(start_idx + 1, end):
        if rows[i].get("class") == klass:
            out.append(rows[i])
    return out


def _patient_from_demographics(rows: list[dict[str, Any]], patient_id: str) -> dict[str, Any] | None:
    """Build one Patient resource from demographic rows, if present."""
    ages = [str(r.get("text", "")).strip() for r in rows if r.get("class") == "demographic_age" and str(r.get("text", "")).strip()]
    sexes = [str(r.get("text", "")).strip().lower() for r in rows if r.get("class") == "demographic_sex" and str(r.get("text", "")).strip()]
    if not ages and not sexes:
        return None

    gender = "unknown"
    for sex in sexes:
        if sex in {"m", "male", "man"}:
            gender = "male"
            break
        if sex in {"f", "female", "woman"}:
            gender = "female"
            break

    patient: dict[str, Any] = {
        "resourceType": "Patient",
        "id": patient_id,
        "gender": gender,
    }
    if ages:
        patient["extension"] = [{"url": "age", "valueString": ages[0]}]
    return patient


def _medication_statements(rows: list[dict[str, Any]], patient_id: str) -> list[dict[str, Any]]:
    """Build MedicationStatement resources grouped by medication_group."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        attrs = r.get("attributes") or {}
        group = attrs.get("medication_group") if isinstance(attrs, dict) else None
        if group:
            grouped.setdefault(str(group), []).append(r)

    resources: list[dict[str, Any]] = []
    for group_name, g_rows in grouped.items():
        by_class: dict[str, list[str]] = {}
        for r in g_rows:
            by_class.setdefault(str(r.get("class", "")), []).append(str(r.get("text", "")))
        med_name = (by_class.get("medication") or [group_name])[0]
        dosage = (by_class.get("dosage") or [""])[0]
        route = (by_class.get("route") or [""])[0]
        frequency = (by_class.get("frequency") or [""])[0]
        duration = (by_class.get("duration") or [""])[0]
        indication = (by_class.get("indication") or [""])[0]
        med_status = (by_class.get("medication_status") or [""])[0]
        dosage_text = " ".join(x for x in [dosage, route, frequency, duration] if x).strip()
        resources.append(
            {
                "resourceType": "MedicationStatement",
                "id": f"med-{_slug(group_name)}",
                "status": "active",
                "subject": {"reference": f"Patient/{patient_id}"},
                "medicationCodeableConcept": {"text": med_name},
                "dosage": [
                    {
                        "text": dosage_text,
                        "route": {"text": route},
                        "timing": {"repeat": {"frequency": 1, "period": 1, "periodUnit": "d"}},
                    }
                ],
                "reasonCode": [{"text": indication}],
                "note": [{"text": f"status: {med_status}"}],
            }
        )
    return resources


def _conditions_from_diagnoses(rows: list[dict[str, Any]], patient_id: str) -> list[dict[str, Any]]:
    """Build Condition resources from diagnosis rows."""
    resources: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        if r.get("class") != "diagnosis":
            continue
        diagnosis_text = str(r.get("text", ""))
        nearby_status_rows = _find_nearby(rows, i, "diagnosis_status", 3)
        nearby_onset_rows = _find_nearby(rows, i, "diagnosis_onset", 3)
        nearby_icd_rows = _find_nearby(rows, i, "diagnosis_icd", 3)

        status_text = (nearby_status_rows[0].get("text", "") if nearby_status_rows else "").strip().lower()
        status_map = {
            "active": "active",
            "resolved": "resolved",
            "history of": "inactive",
        }
        code_status = status_map.get(status_text, "active")
        code: dict[str, Any] = {"text": diagnosis_text}
        if nearby_icd_rows:
            code["coding"] = [
                {
                    "system": "http://hl7.org/fhir/sid/icd-10",
                    "code": str(nearby_icd_rows[0].get("text", "")),
                }
            ]
        condition: dict[str, Any] = {
            "resourceType": "Condition",
            "id": f"cond-{i}",
            "subject": {"reference": f"Patient/{patient_id}"},
            "code": code,
            "clinicalStatus": {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": code_status,
                    }
                ]
            },
        }
        if nearby_onset_rows:
            condition["onsetString"] = str(nearby_onset_rows[0].get("text", ""))
        resources.append(condition)
    return resources


def _conditions_from_symptoms(rows: list[dict[str, Any]], patient_id: str) -> list[dict[str, Any]]:
    """Build symptom-sign Conditions."""
    resources: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        if r.get("class") != "symptom_sign":
            continue
        attrs = r.get("attributes") if isinstance(r.get("attributes"), dict) else {}
        resources.append(
            {
                "resourceType": "Condition",
                "id": f"symptom-{i}",
                "subject": {"reference": f"Patient/{patient_id}"},
                "code": {"text": str(r.get("text", ""))},
                "category": [{"coding": [{"code": "problem-list-item"}]}],
                "clinicalStatus": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                            "code": "active",
                        }
                    ]
                },
                "note": [{"text": f"negated: {attrs.get('negated', 'false')}"}],
            }
        )
    return resources


def _observations(rows: list[dict[str, Any]], patient_id: str) -> list[dict[str, Any]]:
    """Build Observation resources grouped by lab_group."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        attrs = r.get("attributes") or {}
        group = attrs.get("lab_group") if isinstance(attrs, dict) else None
        if group:
            grouped.setdefault(str(group), []).append(r)

    resources: list[dict[str, Any]] = []
    for group_name, g_rows in grouped.items():
        by_class: dict[str, list[str]] = {}
        for r in g_rows:
            by_class.setdefault(str(r.get("class", "")), []).append(str(r.get("text", "")))
        resources.append(
            {
                "resourceType": "Observation",
                "id": f"obs-{_slug(group_name)}",
                "status": "final",
                "subject": {"reference": f"Patient/{patient_id}"},
                "code": {"text": (by_class.get("lab_test") or [""])[0]},
                "valueString": (by_class.get("lab_value") or [""])[0],
                "referenceRange": [{"text": (by_class.get("lab_reference") or [""])[0]}],
                "interpretation": [{"text": (by_class.get("lab_interpretation") or [""])[0]}],
            }
        )
    return resources


def _procedures(rows: list[dict[str, Any]], patient_id: str) -> list[dict[str, Any]]:
    """Build Procedure resources from procedure rows and nearby details."""
    resources: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        if r.get("class") != "procedure":
            continue
        nearby_dates = _find_nearby(rows, i, "procedure_date", 5)
        nearby_findings = _find_nearby(rows, i, "procedure_findings", 5)
        resources.append(
            {
                "resourceType": "Procedure",
                "id": f"proc-{i}",
                "status": "completed",
                "subject": {"reference": f"Patient/{patient_id}"},
                "code": {"text": str(r.get("text", ""))},
                "performedString": str(nearby_dates[0].get("text", "")) if nearby_dates else "",
                "note": [{"text": "; ".join(str(x.get("text", "")) for x in nearby_findings)}],
            }
        )
    return resources


def _allergies(rows: list[dict[str, Any]], patient_id: str) -> list[dict[str, Any]]:
    """Build AllergyIntolerance resources."""
    resources: list[dict[str, Any]] = []
    for i, r in enumerate(rows):
        if r.get("class") != "adverse_event_allergy":
            continue
        text = str(r.get("text", ""))
        resources.append(
            {
                "resourceType": "AllergyIntolerance",
                "id": f"allergy-{i}",
                "patient": {"reference": f"Patient/{patient_id}"},
                "code": {"text": text},
                "reaction": [{"description": text}],
            }
        )
    return resources


def rows_to_fhir_bundle(rows: list[dict[str, Any]], patient_id: str = "patient-1") -> dict[str, Any]:
    """Convert extraction rows into a FHIR R4 Bundle (collection)."""
    resources: list[dict[str, Any]] = []

    patient = _patient_from_demographics(rows, patient_id)
    if patient:
        resources.append(patient)
    resources.extend(_medication_statements(rows, patient_id))
    resources.extend(_conditions_from_diagnoses(rows, patient_id))
    resources.extend(_conditions_from_symptoms(rows, patient_id))
    resources.extend(_observations(rows, patient_id))
    resources.extend(_procedures(rows, patient_id))
    resources.extend(_allergies(rows, patient_id))

    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": r} for r in resources],
    }
