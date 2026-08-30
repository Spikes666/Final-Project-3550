"""Predict a categorical target column and compare classification models."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from datalens.classification import feature_importances, run_classification_comparison
from datalens.eda import categorical_columns
from datalens.plotting import confusion_matrix_plot, metric_bar_chart

st.set_page_config(page_title="Classification — DataLens", page_icon="🏷️", layout="wide")
st.title("🏷️ Classification")

df = st.session_state.get("df")
if df is None:
    st.warning("No dataset loaded yet. Go to the **Home** page and upload one first.")
    st.stop()

candidate_targets = [c for c in categorical_columns(df) if 2 <= df[c].nunique() <= 20]
if not candidate_targets:
    st.error("No suitable categorical column found (need 2–20 distinct values) to classify.")
    st.stop()

target = st.selectbox("Target column to predict", candidate_targets)
candidate_features = [c for c in df.columns if c != target]
features = st.multiselect("Feature columns", candidate_features, default=candidate_features)

if not features:
    st.info("Pick at least one feature column.")
    st.stop()

if st.button("Run models", type="primary"):
    with st.spinner("Training Logistic Regression, Decision Tree, and Random Forest models..."):
        results = run_classification_comparison(df, target, features)
    st.session_state["classification_results"] = results
    st.session_state["classification_target"] = target

results = st.session_state.get("classification_results")
if results and st.session_state.get("classification_target") == target:
    st.subheader("Model comparison")
    comparison = pd.DataFrame(
        [{"Model": r.model_name, "Accuracy": r.accuracy} for r in results]
    ).set_index("Model")
    st.dataframe(comparison.style.format("{:.3f}"), use_container_width=True)
    st.pyplot(
        metric_bar_chart(comparison.index.tolist(), comparison["Accuracy"].tolist(), "Accuracy", "Accuracy by model"),
        use_container_width=True,
    )

    best = max(results, key=lambda r: r.accuracy)
    st.success(f"Best model: **{best.model_name}** (accuracy = {best.accuracy:.3f})")

    st.subheader(f"Confusion matrix — {best.model_name}")
    st.pyplot(confusion_matrix_plot(best.confusion, best.labels), use_container_width=True)

    importances = feature_importances(best)
    if importances is not None:
        st.subheader(f"Feature importance — {best.model_name}")
        st.bar_chart(importances.head(15))
