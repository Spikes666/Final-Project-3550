"""Shared preprocessing pipeline for modeling on arbitrary tabular data."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from datalens.eda import categorical_columns, numeric_columns

# Columns with more distinct values than this are dropped from one-hot
# encoding candidates (e.g. free-text or ID columns would blow up the
# feature space).
MAX_CATEGORY_CARDINALITY = 50


def build_feature_pipeline(df: pd.DataFrame, feature_columns: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build a ColumnTransformer that imputes/scales numeric features and
    imputes/one-hot-encodes categorical features, dropping high-cardinality
    categoricals that aren't useful as model inputs.
    """
    features = df[feature_columns]

    numeric = numeric_columns(features)
    categorical = [
        col
        for col in categorical_columns(features)
        if features[col].nunique(dropna=True) <= MAX_CATEGORY_CARDINALITY
    ]

    transformers = []
    if numeric:
        numeric_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return preprocessor, numeric, categorical
