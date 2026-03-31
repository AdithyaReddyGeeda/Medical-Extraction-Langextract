from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="ClinicalExtract",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

extract_page = st.Page("pages/extract.py", title="Extract", icon="🔬", default=True)
evaluate_page = st.Page("pages/evaluate.py", title="Evaluate", icon="📊")
history_page = st.Page("pages/history.py", title="History", icon="🕓")

pg = st.navigation([extract_page, evaluate_page, history_page])
pg.run()
