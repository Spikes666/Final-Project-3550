"""Predict a numeric target column and compare regression models."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from datalens.eda import numeric_columns
from datalens.plotting import metric_bar_chart
from datalens.regression import feature_importances, run_regression_comparison

st.set_page_config(page_title="Regression — DataLens", page_icon="📈", layout="wide")
st.title("📈 Regression")

df = st.session_state.get("df")
if df is None:
    st.warning("No dataset loaded yet. Go to the **Home** page and upload one first.")
    st.stop()

targets = numeric_columns(df)
if not targets:
    st.error("This dataset has no numeric columns, so there's no target to regress on.")
    st.stop()

target = st.selectbox("Target column to predict", targets)
candidate_features = [c for c in df.columns if c != target]
features = st.multiselect("Feature columns", candidate_features, default=candidate_features)

if not features:
    st.info("Pick at least one feature column.")
    st.stop()

if st.button("Run models", type="primary"):
    with st.spinner("Training Linear, LASSO, Ridge, and Random Forest models..."):
        results = run_regression_comparison(df, target, features)
    st.session_state["regression_results"] = results
    st.session_state["regression_target"] = target
    st.session_state["regression_features"] = features

results = st.session_state.get("regression_results")
if results and st.session_state.get("regression_target") == target:
    st.subheader("Model comparison")
    comparison = pd.DataFrame(
        [{"Model": r.model_name, "R²": r.r2, "MAE": r.mae, "RMSE": r.rmse} for r in results]
    ).set_index("Model")
    st.dataframe(comparison.style.format("{:.3f}"), use_container_width=True)
    st.pyplot(
        metric_bar_chart(comparison.index.tolist(), comparison["R²"].tolist(), "R²", "R² by model"),
        use_container_width=True,
    )

    best = max(results, key=lambda r: r.r2)
    st.success(f"Best model: **{best.model_name}** (R² = {best.r2:.3f})")

    importances = feature_importances(best)
    if importances is not None:
        st.subheader(f"Feature importance — {best.model_name}")
        st.bar_chart(importances.head(15))
