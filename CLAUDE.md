# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

DataLens is a dataset-agnostic statistical analysis framework, built as a Streamlit app.
A user uploads any tabular dataset (CSV/TSV/Excel/JSON/Parquet) through the browser and
gets exploratory analysis, regression, classification, and clustering — with no dataset-
specific code required. This replaced an earlier, single-purpose coursework notebook; there
is no longer any project-specific data or modeling logic baked into the codebase — all of
it works generically off whatever columns the uploaded dataset has.

## Commands

```bash
pip install -r requirements-dev.txt   # app deps + pytest
streamlit run app.py                  # run the app locally (http://localhost:8501)
pytest                                # run the test suite
pytest tests/test_modeling.py -k regression   # run a single test
docker build -t datalens . && docker run -p 8501:8501 datalens   # containerized run
```

There is no linter/formatter configured. `conftest.py` at the repo root is empty — it
exists solely so pytest adds the repo root to `sys.path`, making `datalens.*` importable
from `tests/`.

## Architecture

The codebase is split into three layers:

1. **`datalens/`** — the analysis engine, pure functions/dataclasses with no Streamlit
   dependency (aside from not importing it at all). This is where new analysis
   capabilities belong, and it's unit-testable in isolation:
   - `loading.py` — dispatches on file extension to load a DataFrame from an uploaded
     file-like object or path (`SUPPORTED_EXTENSIONS`).
   - `eda.py` — dtype/missingness/cardinality overview, numeric/categorical column
     splitting, summary statistics, correlation matrix. Every other module's "which
     columns can I use for X" logic goes through `numeric_columns`/`categorical_columns`
     here.
   - `preprocessing.py` — `build_feature_pipeline` builds a `sklearn.ColumnTransformer`
     (median-impute + scale numeric, most-frequent-impute + one-hot categorical) shared by
     regression, classification, and clustering. High-cardinality categoricals (>
     `MAX_CATEGORY_CARDINALITY`) are dropped automatically so ID-like columns don't blow up
     the one-hot feature space.
   - `regression.py` / `classification.py` — each fits every model in its `MODELS` dict
     inside a fresh `Pipeline([("preprocess", ...), ("model", ...)])`, returning a list of
     result dataclasses (`RegressionResult`/`ClassificationResult`) plus a
     `feature_importances()` helper that reads `model.feature_importances_` through the
     fitted preprocessor's `get_feature_names_out()`.
   - `clustering.py` — k-means over the same shared preprocessing pipeline, plus a PCA
     projection for plotting and an elbow-curve helper.
   - `plotting.py` — the only place matplotlib figures are built; pages call these instead
     of embedding plotting code.

2. **`app.py` + `pages/`** — the Streamlit UI. `app.py` is the Home page: it's the only
   place a dataset is loaded, and it stores the result in `st.session_state["df"]` (and
   `dataset_name`). Every page under `pages/` (numbered so Streamlit's multipage nav orders
   them: `1_Explore.py`, `2_Regression.py`, `3_Classification.py`, `4_Clustering.py`) starts
   by reading `st.session_state.get("df")` and stopping with a warning if it's `None` —
   there is no other way pages get data. Pages are thin: they collect widget input (target/
   feature column pickers), call one function from `datalens/`, and render the result.
   Regression/Classification/Clustering pages cache their last run in
   `st.session_state[f"{page}_results"]` keyed alongside the chosen target/k, so switching
   pages and coming back doesn't silently show stale results for a different selection.

3. **Deployment config** — `.streamlit/config.toml` (theme, upload size limit, CORS/XSRF),
   `Dockerfile` (installs `requirements.txt`, runs `streamlit run app.py` on `0.0.0.0:8501`),
   and `requirements.txt`/`requirements-dev.txt`.

## Conventions

- Keep `datalens/` free of Streamlit imports so its functions stay unit-testable without a
  running app (see `tests/test_modeling.py` for the pattern: build a small synthetic mixed
  numeric/categorical DataFrame, run the comparison function, assert on shape/ranges rather
  than exact scores).
- New model types go into the relevant module's `MODELS` dict, not into the Streamlit page
  — pages iterate over whatever `run_*_comparison` returns.
- Any new column-selection logic (e.g. "which columns are valid targets") should reuse or
  extend `datalens/eda.py`'s helpers rather than re-implementing dtype checks in a page.
- `sample_data/sample_housing.csv` is a synthetic dataset (see git history for the
  generator script) used by the Home page's "Use sample dataset" button and by nothing
  else — it's not read by any test.
