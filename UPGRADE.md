# ClinicalExtract — Upgrade Guide

This document outlines how to upgrade and improve the ClinicalExtract project (medical information extraction with LangExtract).

---

## 1. Dependencies

| Current | Recommendation | Notes |
|--------|-----------------|--------|
| `langextract>=1.1.0,<2.0.0` | `langextract>=1.1.1,<2.0.0` | Pin to latest 1.x (1.1.1) for fixes. |
| `streamlit>=1.38.0` | `streamlit>=1.40.0` | Newer Streamlit for better UX and security. |
| `pandas>=2.0.0` | Keep or `pandas>=2.2.0` | Optional: align with LangExtract’s pandas>=1.3.0. |
| — | Optional: `langextract[openai]` | Add if you want GPT-4o / OpenAI in the UI. |

**Action:** Update `requirements.txt` and run `pip install -r requirements.txt` (or use a venv).

---

## 2. Environment & API Keys

- **Add `.env` loading in the app** so `LANGEXTRACT_API_KEY` in a `.env` file is picked up without manual `export`. Use `python-dotenv`: call `load_dotenv()` at the very start of `app.py` (and optionally in `extractor.py` if run as a script).
- **Add `.env.example`** with:
  - `LANGEXTRACT_API_KEY=your-gemini-api-key`
  - Optional: `OPENAI_API_KEY=...` if you add OpenAI support.

---

## 3. Schema & Extraction (from README “Limitations & Future Work”)

- **Negation / temporality:** You already have `diagnosis_status`, `medication_status`, and a `negated` attribute on symptoms. To upgrade:
  - Add few-shot examples that explicitly show “denied”, “history of”, “resolved”, “no longer”.
  - Optionally add entity types or attributes like `temporality` (e.g. past, current, resolved).
- **Relation extraction:** Medication→indication is already modeled via `medication_group` and `indication`. Possible upgrades:
  - Add an “Relations” or “Evidence” view in the UI that groups by `medication_group` / `lab_group`.
  - Optional: export a simple graph (e.g. JSON of relations) for downstream use.
- **Languages:** Keep English-focused; add a short note in README that prompts/examples are English-only and that multi-language would require translated prompts and examples.

---

## 4. Streamlit App & UX

- **Caching:** Use `@st.cache_data` (with a hash of input text + model settings) for the extraction result so re-running with the same input doesn’t call the LLM again. Invalidate when text or key settings change.
- **File types:** Currently only `.txt`. Consider:
  - **PDF:** Use `pypdf` or `pdfplumber` to extract text, then run the same pipeline.
  - **DOCX:** Use `python-docx` to extract text.
- **Batch upload:** Allow multiple files; run extraction per file and show a list of results (e.g. per-file table or combined export).
- **Providers:** Add “OpenAI (cloud)” next to Ollama and Gemini in the sidebar if you add `langextract[openai]`; pass `model_id="gpt-4o"`, `fence_output=True`, `use_schema_constraints=False` as in LangExtract docs.

---

## 5. Evaluation

- **One-command eval:** The eval script expects `*_pred.json` next to each sample. Add a CLI or script that:
  1. Runs the extractor on each `samples/*.txt`,
  2. Writes `samples/<name>_pred.json`,
  3. Runs the existing eval (e.g. `python -m utils.eval --samples samples/ --output eval_results/`).
- **Per-entity metrics:** Extend `utils/eval.py` to report precision/recall/F1 per extraction class (e.g. medication, diagnosis, lab_test) in addition to aggregate.

---

## 6. Production & Scale (Optional)

- **Vertex AI:** For production, use Vertex AI with service account auth and optional **batch API** to reduce cost on large volumes. See LangExtract docs: `language_model_params={"vertexai": True, "batch": {"enabled": True}}`.
- **Docker:** Add a `Dockerfile` that installs deps and runs `streamlit run app.py` for consistent deployments.

---

## 7. Code Quality & Packaging

- **Type hints:** Already good; ensure any new functions stay typed.
- **Tests:** Add a small `tests/` with pytest:
  - Test `extractions_to_serializable()` with a mock `AnnotatedDocument`.
  - Test eval `compute_metrics()` with fixed pred/gold lists.
- **Packaging:** Optional `pyproject.toml` for install as a package and modern dependency spec (e.g. `pip install -e .`).

---

## 8. Documentation

- **Demo:** Add a screenshot or short GIF of the Streamlit app to README (replace the TODO).
- **README:** Add a one-line “Upgrade” section that points to this file.

---

## Quick wins (minimal code changes)

1. Bump `langextract` to `>=1.1.1` and `streamlit` to `>=1.40.0` in `requirements.txt`.
2. Call `load_dotenv()` at the top of `app.py` (and optionally in `extractor.py`).
3. Add `.env.example` with `LANGEXTRACT_API_KEY=...`.
4. Add a short “Upgrade” section in README linking to `UPGRADE.md`.

After that, tackle schema/UX/eval items in order of priority for your use case.
