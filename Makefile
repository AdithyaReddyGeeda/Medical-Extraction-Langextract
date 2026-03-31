PYTHON ?= python3

.PHONY: test eval run

test:
	$(PYTHON) -m pytest -q

eval:
	$(PYTHON) evaluate.py --provider gemini --model gemini-2.5-flash

run:
	streamlit run app.py
