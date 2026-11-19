# Diabetes Across Ages

**Age-stratified diabetes analytics and machine learning using NHANES 2021–2023 public-use data.**

## What is delivered

- Interactive Streamlit application (`app.py`) with overview, child BMI-for-age, adult, older-adult, model-evaluation, prediction-demo and methodology sections.
- Original NHANES XPT files used by the analysis (`data/`).
- Reproducible model development pipeline (`diabetes_ml_validation_pipeline.py`).
- SHAP helper (`add_shap.py`).
- Saved evaluation outputs (`results/`).
- Editable report draft (`docs/Project_Report_Draft.docx`), methodology/data dictionary, deployment guide and viva preparation notes.

## Key result

The selected adult model is Logistic Regression. On the held-out test set (n=1,576), at validation-selected threshold 0.64: accuracy 78.30%, precision 32.31%, recall 53.95%, specificity 82.15%, F1 0.404 and ROC-AUC 0.788.

## Important scope

- Adult classifier predicts self-reported diagnosed-diabetes label, not future onset.
- Children are described using BMI-for-age categories only; no child diabetes prevalence or child classifier is claimed.
- All estimates and model metrics are unweighted and not nationally representative.
- The application is interactive but uses static public survey files. It does not connect to live clinical data.
- It is not a clinical screening or decision-support tool.
