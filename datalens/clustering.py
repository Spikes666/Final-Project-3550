"""Unsupervised clustering for datasets without a chosen target column."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from datalens.preprocessing import build_feature_pipeline


@dataclass
class ClusteringResult:
    k: int
    labels: np.ndarray
    inertia: float
    silhouette: float | None
    projection: np.ndarray  # 2D PCA projection for plotting


def run_clustering(
    df: pd.DataFrame,
    feature_columns: list[str],
    k: int,
    random_state: int = 0,
) -> ClusteringResult:
    data = df[feature_columns].dropna()
    preprocessor, _, _ = build_feature_pipeline(data, feature_columns)
    X = preprocessor.fit_transform(data)
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)

    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)

    silhouette = silhouette_score(X, labels) if k > 1 and len(set(labels)) > 1 else None

    n_components = min(2, X.shape[1])
    projection = PCA(n_components=n_components, random_state=random_state).fit_transform(X)
    if n_components == 1:
        projection = np.column_stack([projection, np.zeros(len(projection))])

    return ClusteringResult(
        k=k, labels=labels, inertia=model.inertia_, silhouette=silhouette, projection=projection
    )


def elbow_curve(df: pd.DataFrame, feature_columns: list[str], k_range: range, random_state: int = 0) -> pd.DataFrame:
    data = df[feature_columns].dropna()
    preprocessor, _, _ = build_feature_pipeline(data, feature_columns)
    X = preprocessor.fit_transform(data)
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)

    rows = []
    for k in k_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
        rows.append({"k": k, "inertia": model.inertia_})
    return pd.DataFrame(rows)
