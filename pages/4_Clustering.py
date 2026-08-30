"""Group similar rows together when there's no target column to predict."""

from __future__ import annotations

import streamlit as st

from datalens.clustering import elbow_curve, run_clustering
from datalens.eda import numeric_columns
from datalens.plotting import scatter_projection

st.set_page_config(page_title="Clustering — DataLens", page_icon="🧩", layout="wide")
st.title("🧩 Clustering")

df = st.session_state.get("df")
if df is None:
    st.warning("No dataset loaded yet. Go to the **Home** page and upload one first.")
    st.stop()

candidate_features = numeric_columns(df) + [c for c in df.columns if df[c].nunique() <= 50]
candidate_features = list(dict.fromkeys(candidate_features))  # de-dupe, keep order
features = st.multiselect("Feature columns", candidate_features, default=numeric_columns(df))

if len(features) < 1:
    st.info("Pick at least one feature column.")
    st.stop()

max_k = min(10, max(2, df[features].dropna().shape[0] - 1))
k = st.slider("Number of clusters (k)", min_value=2, max_value=max_k, value=min(3, max_k))

if st.button("Run clustering", type="primary"):
    with st.spinner("Clustering rows and computing the elbow curve..."):
        result = run_clustering(df, features, k)
        elbow = elbow_curve(df, features, range(2, max_k + 1))
    st.session_state["clustering_result"] = result
    st.session_state["clustering_elbow"] = elbow
    st.session_state["clustering_k"] = k

result = st.session_state.get("clustering_result")
if result and st.session_state.get("clustering_k") == k:
    col1, col2 = st.columns(2)
    col1.metric("Inertia", f"{result.inertia:,.1f}")
    col2.metric("Silhouette score", f"{result.silhouette:.3f}" if result.silhouette is not None else "n/a")

    st.subheader("Cluster projection (PCA)")
    st.pyplot(
        scatter_projection(result.projection, result.labels, f"k={result.k} clusters"),
        use_container_width=True,
    )

    st.subheader("Elbow curve")
    elbow = st.session_state["clustering_elbow"]
    st.line_chart(elbow.set_index("k"))
