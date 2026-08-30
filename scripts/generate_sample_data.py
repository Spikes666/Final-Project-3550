"""Regenerate sample_data/sample_housing.csv, a synthetic dataset used by the
Home page's "Use sample dataset" button. Not used by tests or the app at
runtime otherwise.
"""

import numpy as np
import pandas as pd

OUTPUT_PATH = "sample_data/sample_housing.csv"


def generate(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    sqft = rng.normal(1800, 500, n).clip(400, 4000)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = rng.integers(1, 4, n)
    age = rng.integers(0, 80, n)
    neighborhood = rng.choice(["Downtown", "Suburb", "Rural", "Waterfront"], n, p=[0.3, 0.4, 0.2, 0.1])
    garage = rng.choice(["Yes", "No"], n, p=[0.6, 0.4])

    neighborhood_premium = {"Downtown": 60000, "Suburb": 20000, "Rural": -10000, "Waterfront": 90000}
    price = (
        sqft * 150
        + bedrooms * 8000
        + bathrooms * 6000
        - age * 500
        + np.array([neighborhood_premium[n_] for n_ in neighborhood])
        + (garage == "Yes") * 12000
        + rng.normal(0, 25000, n)
    ).round(-2)

    risk_score = (
        (age > 40).astype(int)
        + (sqft < 1200).astype(int)
        + (neighborhood == "Rural").astype(int)
        + rng.integers(0, 2, n)
    )
    risk_label = np.where(risk_score >= 2, "High", "Low")

    # Sprinkle a few missing values so the app's imputation path gets exercised.
    sqft = sqft.copy()
    missing_idx = rng.choice(n, size=8, replace=False)
    sqft[missing_idx] = np.nan

    return pd.DataFrame(
        {
            "sqft": sqft.round(0),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age,
            "neighborhood": neighborhood,
            "has_garage": garage,
            "sale_price": price,
            "risk_label": risk_label,
        }
    )


if __name__ == "__main__":
    generate().to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")
