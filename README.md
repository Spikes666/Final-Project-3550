# DataLens

A dataset-agnostic statistical analysis framework. Upload a CSV, TSV, Excel,
JSON, or Parquet file and DataLens explores it, fits and compares regression,
classification, or clustering models, and shows the results — all from a
mobile-friendly Streamlit UI.

## Features

- **Any tabular dataset** — CSV, TSV, XLSX/XLS, JSON, or Parquet, uploaded through the browser (no local file paths to edit).
- **Explore** — column overview, missing-value summary, distributions, correlation heatmap.
- **Regression** — pick a numeric target and compare Linear, LASSO, Ridge, and Random Forest models on R², MAE, and RMSE.
- **Classification** — pick a categorical target and compare Logistic Regression, Decision Tree, and Random Forest models on accuracy and a confusion matrix.
- **Clustering** — group rows with k-means when there's no target column, with an elbow curve and a PCA scatter plot.
- **Mobile-friendly** — Streamlit's responsive layout collapses the sidebar into a menu on small screens.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the printed local URL, upload a dataset (or click **Use sample
dataset** to try it with a bundled synthetic housing dataset), and use the
sidebar to navigate between Explore / Regression / Classification /
Clustering.

## Running with Docker

```bash
docker build -t datalens .
docker run -p 8501:8501 datalens
```

## Deploying

- **Streamlit Community Cloud**: point it at this repo with `app.py` as the
  entrypoint; `requirements.txt` and `.streamlit/config.toml` are already set
  up for it.
- **Any Docker host**: build and run the image above behind your usual
  reverse proxy / TLS termination.

## Development

```bash
pip install -r requirements-dev.txt
pytest
```
