# Diabetes Across Ages: interactive ML dashboard

Age-stratified diabetes analytics and an explainable machine-learning classifier on
NHANES August 2021 - August 2023 public-use data (DEMO_L, DIQ_L, BMX_L, PAQ_L).

Academic project (KIIT, School of Computer Science and Engineering). Not a medical tool.

## Run on Windows (local, no server needed)

1. Install **Python 3.10 or newer** from python.org (tick "Add Python to PATH").
2. Extract this ZIP completely.
3. Open **Command Prompt** inside the extracted `Diabetes_Across_Ages_Final_Project` folder
   (click the folder's address bar, type `cmd`, press Enter).
4. Run these commands one by one:

```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

5. Your browser opens at http://localhost:8501. If it does not, open that address yourself.
6. To stop the app press `Ctrl + C` in the Command Prompt window.

Next time, you only need steps 3, then `.venv\Scripts\activate`, then `streamlit run app.py`.

## Dashboard sections

1. Overview
2. Diabetes Across Age Groups (age-group, age-range and sex filters)
3. Machine Learning Explorer (Logistic Regression, Random Forest, XGBoost, selection method)
4. Model Performance (validation and held-out test results shown separately)
5. Interactive ML Prediction (live Logistic Regression pipeline)
6. Model Explainability (SHAP and permutation importance)
7. Dataset & Methodology (includes a reproducibility check)

## How the prediction page works

`core.py` rebuilds the exact experiment pipeline (same cleaning, imputation, scaling, one-hot encoding,
and the same 60/20/20 split seeds 42 and 43) and trains Logistic Regression on the **training split only**.
Your inputs go through the same recoding and pipeline. The "Reproducibility check" on the
Dataset & Methodology page re-scores the held-out test set and compares it with `results/`.
In our test it matched the saved results exactly (ROC-AUC 0.7883, confusion matrix 1118 / 243 / 99 / 116).

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit dashboard |
| `core.py` | Data preparation, pipeline, prediction, curves (no UI code) |
| `diabetes_ml_validation_pipeline.py` | Original experiment: model comparison and final test (needs xgboost) |
| `add_shap.py` | SHAP importance script |
| `results/` | Saved validation, test, age-wise, permutation and SHAP results |
| `data/` | The four NHANES XPT files |
| `presentation/` | PowerPoint (.pptx) and PDF |
| `docs/DEMO_SCRIPT.md` | What to show your professor |
| `tests/smoke_test_logic.py` | Logic smoke test (see below) |

## Optional: regenerate the experiment results

```
python diabetes_ml_validation_pipeline.py
python add_shap.py
```

## Optional: logic smoke test

`python tests/smoke_test_logic.py` runs every dashboard page with stand-in Streamlit/Plotly objects
to check the data and model code. It does not test visual rendering.

## Scope reminders

- The classifier is trained for **adults 18+** only; there is **no** child diabetes classifier.
- All proportions and metrics are **unweighted** and are **not** national prevalence estimates.
- Outcome is **self-reported** diagnosed diabetes (DIQ010). One survey cycle, no external validation.
- The 0.64 threshold maximises validation F1; it is not a medically recommended threshold.
