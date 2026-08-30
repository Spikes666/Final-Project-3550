"""Exploratory data analysis for whatever dataset is loaded on the Home page."""

from __future__ import annotations

import streamlit as st

from datalens.eda import (
    categorical_columns,
    column_overview,
    correlation_matrix,
    numeric_columns,
    summary_statistics,
)
from datalens.plotting import correlation_heatmap

st.set_page_config(page_title="Explore — DataLens", page_icon="📊", layout="wide")
st.title("📊 Explore")

df = st.session_state.get("df")
if df is None:
    st.warning("No dataset loaded yet. Go to the **Home** page and upload one first.")
    st.stop()

st.subheader("Column overview")
st.dataframe(column_overview(df), use_container_width=True)

numeric_cols = numeric_columns(df)
categorical_cols = categorical_columns(df)

st.subheader("Summary statistics (numeric columns)")
stats = summary_statistics(df)
if stats.empty:
    st.info("No numeric columns to summarize.")
else:
    st.dataframe(stats, use_container_width=True)

st.subheader("Distribution")
if numeric_cols:
    col = st.selectbox("Numeric column", numeric_cols, key="dist_numeric")
    st.bar_chart(df[col].dropna().value_counts(bins=20).sort_index())
elif categorical_cols:
    col = st.selectbox("Categorical column", categorical_cols, key="dist_categorical")
    st.bar_chart(df[col].value_counts())
else:
    st.info("No columns available to plot.")

st.subheader("Correlation heatmap")
corr = correlation_matrix(df)
if corr.empty:
    st.info("Need at least two numeric columns for a correlation heatmap.")
else:
    st.pyplot(correlation_heatmap(corr), use_container_width=True)
