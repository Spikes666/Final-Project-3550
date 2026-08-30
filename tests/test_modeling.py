import numpy as np
import pandas as pd
import pytest

from datalens.classification import run_classification_comparison
from datalens.clustering import run_clustering
from datalens.regression import run_regression_comparison


@pytest.fixture
def mixed_df():
    rng = np.random.default_rng(0)
    n = 60
    x1 = rng.normal(size=n)
    x2 = rng.choice(["a", "b", "c"], size=n)
    target_num = x1 * 3 + (x2 == "a").astype(float) * 2 + rng.normal(scale=0.1, size=n)
    target_cat = np.where(target_num > np.median(target_num), "high", "low")
    return pd.DataFrame({"x1": x1, "x2": x2, "target_num": target_num, "target_cat": target_cat})


def test_regression_comparison_runs(mixed_df):
    results = run_regression_comparison(mixed_df, "target_num", ["x1", "x2"])

    assert len(results) == 4
    assert all(r.r2 <= 1.0 for r in results)


def test_classification_comparison_runs(mixed_df):
    results = run_classification_comparison(mixed_df, "target_cat", ["x1", "x2"])

    assert len(results) == 3
    assert all(0.0 <= r.accuracy <= 1.0 for r in results)


def test_clustering_runs(mixed_df):
    result = run_clustering(mixed_df, ["x1"], k=2)

    assert result.k == 2
    assert len(result.labels) == len(mixed_df)
    assert result.projection.shape == (len(mixed_df), 2)
