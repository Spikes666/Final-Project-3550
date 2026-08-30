"""Fit and compare regression models on a numeric target."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from datalens.preprocessing import build_feature_pipeline

MODELS = {
    "Linear Regression": LinearRegression(),
    "LASSO Regression": Lasso(alpha=0.1, random_state=0),
    "Ridge Regression": Ridge(alpha=1.0, random_state=0),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=0),
}


@dataclass
class RegressionResult:
    model_name: str
    pipeline: Pipeline
    r2: float
    mae: float
    rmse: float
    y_test: np.ndarray
    y_pred: np.ndarray


def run_regression_comparison(
    df: pd.DataFrame,
    target: str,
    feature_columns: list[str],
    test_size: float = 0.2,
    random_state: int = 0,
) -> list[RegressionResult]:
    data = df.dropna(subset=[target])
    X = data[feature_columns]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    results = []
    for name, estimator in MODELS.items():
        preprocessor, _, _ = build_feature_pipeline(data, feature_columns)
        pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        results.append(
            RegressionResult(
                model_name=name,
                pipeline=pipeline,
                r2=r2_score(y_test, y_pred),
                mae=mean_absolute_error(y_test, y_pred),
                rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
                y_test=y_test.to_numpy(),
                y_pred=y_pred,
            )
        )
    return results


def feature_importances(result: RegressionResult) -> pd.Series | None:
    model = result.pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    preprocessor = result.pipeline.named_steps["preprocess"]
    names = list(preprocessor.get_feature_names_out())
    return pd.Series(model.feature_importances_, index=names).sort_values(ascending=False)
