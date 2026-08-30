"""Dataset-agnostic exploratory data analysis helpers."""

from __future__ import annotations

import pandas as pd


def column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column dtype, missingness, and cardinality summary."""
    overview = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_pct": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(),
        }
    )
    overview.index.name = "column"
    return overview.reset_index()


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(exclude="number").columns.tolist()


def summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """`describe()` over numeric columns, transposed for display."""
    cols = numeric_columns(df)
    if not cols:
        return pd.DataFrame()
    return df[cols].describe().T


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = numeric_columns(df)
    if len(cols) < 2:
        return pd.DataFrame()
    return df[cols].corr(numeric_only=True)
