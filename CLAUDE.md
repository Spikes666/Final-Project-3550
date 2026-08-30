# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This is a DATA 3550 (Applied Predictive Modeling) final project. It is a data science
coursework submission, not a software application — there is no build system, package
manifest, test suite, or CI. The repository contains two Jupyter notebooks:

- **`DATA3550 Fall 2023 Final Project.ipynb`** — the instructor-provided assignment
  template/prompt. It describes the grading rubric, the required datasets, and the
  deliverables. Treat this as the spec, not as code to modify.
- **`Pr0Fin_SartinoN.ipynb`** — the actual submission. All real analysis, modeling, and
  results live here. This is the file to read/edit for any substantive work.

The assignment framing: acting as an analyst for an insurance company, use three datasets
to (1) predict `MQuotedTotalPayment` (a quote amount) via regression, and (2) classify
policy risk via a derived risk label, then compare model performance.

## Data

The notebooks read CSVs from a local `Dataset/` directory that is **not checked into this
repo** (no data files are committed):

- `Dataset/df_AP02.csv` — Abilitech dataset
- `Dataset/MasterQuote.csv` — quote data (source of the regression target `MQuotedTotalPayment`)
- `Dataset/TheGeneral.csv` — policy/claims data (source of the classification target, an
  engineered risk label)

Any environment working with this notebook needs these three CSVs placed in a `Dataset/`
subdirectory relative to the notebook before cells can execute top-to-bottom.

## Running the notebook

There is no requirements.txt/environment.yml. Dependencies are installed inline via
commented `!pip install` lines near the top of `Pr0Fin_SartinoN.ipynb` (scikit-learn,
missingno, seaborn, ydata-profiling) plus numpy, pandas, matplotlib, statsmodels, and
scipy. Run with Jupyter/JupyterLab; execute cells in order since later sections depend on
DataFrames created earlier (data loading → preprocessing → modeling sections each assume
prior cells already ran).

## Structure of `Pr0Fin_SartinoN.ipynb`

The notebook is organized into sequential sections (by markdown headers), each building on
DataFrames from the previous one:

1. **Installs and Imports** — dependency setup.
2. **Loading and Data Transformation** — one subsection per dataset (Abilitech,
   MasterQuote, TheGeneral), each with its own Loading → Exploratory Data Analysis → Data
   Preprocessing flow (cleaning, imputation, dummy variables, `LabelEncoder`). This is by
   far the largest part of the notebook (cells ~8–95).
3. **Regression Models** (target: `MQuotedTotalPayment`) — Linear Regression, LASSO,
   Ridge, Random Forest Regressor, each in its own subsection (fit → evaluate via
   R², MSE/MAE, etc.).
4. **Classification Models** (target: an engineered risk label) — Logistic Regression,
   Decision Tree, Random Forest Classifier (the latter tuned via `GridSearchCV`, e.g.
   `{'criterion': 'entropy', 'max_depth': None, 'max_features': 'sqrt', 'n_estimators': 100}`).
5. **Results and Model Comparison** — bar charts comparing R² across regressors and
   accuracy across classifiers, at the very end of the notebook.

When modifying modeling code, keep this section order intact and make sure preceding
data-preprocessing cells still produce the columns a downstream model expects — the
notebook is written to run linearly, not as independent, re-orderable cells.

## Conventions

- Sections follow the rubric in the template notebook: a markdown heading + brief summary
  precedes each code cell, per the "Formatting Guidelines" section of
  `DATA3550 Fall 2023 Final Project.ipynb`.
- Feature engineering/cleaning is done separately per dataset before any merging; keep new
  preprocessing scoped to the relevant dataset's subsection rather than in a shared/global
  cell.
