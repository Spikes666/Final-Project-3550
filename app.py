"""DataLens — a dataset-agnostic statistical analysis framework.

Home page: upload a dataset (or load the bundled sample) and preview it.
Use the sidebar pages to explore it, run regression/classification, or
cluster it.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from datalens.loading import load_dataframe

SAMPLE_PATH = "sample_data/sample_housing.csv"

st.set_page_config(
    page_title="DataLens",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="auto",
)

st.title("🔎 DataLens")
st.caption("Upload any tabular dataset, then explore it, model it, and compare results — from your phone or your desktop.")

with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload a dataset",
            type=["csv", "tsv", "xlsx", "xls", "json", "parquet"],
            help="CSV, TSV, Excel, JSON, or Parquet.",
        )
    with col2:
        st.write("")
        st.write("")
        use_sample = st.button("Use sample dataset", use_container_width=True)

if uploaded is not None:
    try:
        df = load_dataframe(uploaded)
        st.session_state["df"] = df
        st.session_state["dataset_name"] = uploaded.name
    except Exception as exc:  # noqa: BLE001 - surface any parse error to the user
        st.error(f"Couldn't read that file: {exc}")

if use_sample:
    df = load_dataframe(SAMPLE_PATH)
    st.session_state["df"] = df
    st.session_state["dataset_name"] = "sample_housing.csv"

df: pd.DataFrame | None = st.session_state.get("df")

if df is None:
    st.info("Upload a dataset above, or click **Use sample dataset** to try DataLens with a synthetic housing dataset.")
    st.stop()

st.success(f"Loaded **{st.session_state.get('dataset_name')}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
st.dataframe(df.head(50), use_container_width=True)

st.divider()
st.markdown(
    """
    **Next steps** — use the sidebar (tap ☰ on mobile) to:
    - **Explore** — summary statistics, missing data, distributions, correlations
    - **Regression** — predict a numeric column and compare models
    - **Classification** — predict a categorical column and compare models
    - **Clustering** — group similar rows when there's no target column
    """
)
