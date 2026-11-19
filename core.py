"""
Core (UI-independent) logic for the Diabetes Across Ages project.

Everything here mirrors diabetes_ml_validation_pipeline.py exactly (same cleaning,
same preprocessing, same 60/20/20 split seeds), so the dashboard's live Logistic
Regression pipeline is the SAME model that produced results/final_heldout_test_metrics.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, brier_score_loss,
                             balanced_accuracy_score, roc_curve, precision_recall_curve)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"

FEATURES = ["RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDFMPIR", "BMXBMI", "PAD680"]
NUMERIC = ["RIDAGEYR", "INDFMPIR", "BMXBMI", "PAD680"]
CATEGORICAL = ["RIAGENDR", "RIDRETH3", "DMDEDUC2"]
THRESHOLD = 0.64  # validation-selected; verified against results/final_heldout_test_metrics.csv

FEATURE_LABELS = {
    "RIDAGEYR": "Age", "RIAGENDR": "Sex", "RIDRETH3": "Race/ethnicity", "DMDEDUC2": "Education",
    "INDFMPIR": "Income-to-poverty ratio", "BMXBMI": "BMI", "PAD680": "Sedentary time",
}
SEX_LABELS = {1: "Male", 2: "Female"}
RACE_LABELS = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White", 4: "Non-Hispanic Black",
               6: "Non-Hispanic Asian", 7: "Other race, including multi-racial"}
EDU_LABELS = {1: "Less than 9th grade", 2: "9th-11th grade (no diploma)", 3: "High school graduate / GED",
              4: "Some college or associate degree", 5: "College graduate or above"}
AGE_GROUP_YOUNG = "Adults 18–59"
AGE_GROUP_OLD = "Older adults 60+"


# ----------------------------------------------------------------------------- data
def _read_xpt(stem):
    return pd.read_sas(DATA / f"{stem}.xpt", format="xport")


def load_sources():
    """Return DEMO_L, DIQ_L, BMX_L, PAQ_L and the SEQN-merged table."""
    demo, diq, bmx, paq = [_read_xpt(s) for s in ["DEMO_L", "DIQ_L", "BMX_L", "PAQ_L"]]
    merged = (demo.merge(diq, on="SEQN", how="left", validate="one_to_one")
                  .merge(bmx, on="SEQN", how="left", validate="one_to_one")
                  .merge(paq, on="SEQN", how="left", validate="one_to_one"))
    return demo, diq, bmx, paq, merged


def clean_predictors(df):
    """Identical recoding to the training pipeline (applied to training data AND to app inputs)."""
    df = df.copy()
    for c in ["DMDEDUC2", "RIAGENDR", "RIDRETH3"]:
        df[c] = df[c].replace({7: np.nan, 9: np.nan})
    df["PAD680"] = df["PAD680"].replace({7777: np.nan, 9999: np.nan})
    return df


def adult_frame(merged):
    """Eligible adults (age >= 18, DIQ010 in {1, 2}) with RAW coded columns plus labelled helper columns."""
    df = merged[(merged.RIDAGEYR >= 18) & (merged.DIQ010.isin([1, 2]))].copy()
    df["target"] = (df.DIQ010 == 1).astype(int)
    df["age_group"] = np.where(df.RIDAGEYR < 60, AGE_GROUP_YOUNG, AGE_GROUP_OLD)
    df["age_band"] = pd.cut(df.RIDAGEYR, bins=[18, 30, 40, 50, 60, 70, 80, 200], right=False,
                            labels=["18–29", "30–39", "40–49", "50–59", "60–69", "70–79", "80+"])
    df["bmi_category"] = pd.cut(df.BMXBMI, bins=[0, 18.5, 25, 30, 35, 200], right=False,
                                labels=["Under 18.5", "18.5–24.9", "25–29.9", "30–34.9", "35 and above"])
    df["sex_label"] = df.RIAGENDR.map(SEX_LABELS)
    df["race_label"] = df.RIDRETH3.map(RACE_LABELS)
    df["edu_label"] = df.DMDEDUC2.map(EDU_LABELS)
    sit = df.PAD680.where(~df.PAD680.isin([7777, 9999]))
    df["sitting_hours"] = sit / 60.0
    df["sitting_band"] = pd.cut(df.sitting_hours, bins=[-0.01, 4, 6, 8, 10, 30], labels=["≤4 h", "4–6 h", "6–8 h", "8–10 h", ">10 h"])
    df["pir_band"] = pd.cut(df.INDFMPIR, bins=[-0.01, 1.0, 2.0, 3.0, 4.0, 5.01],
                            labels=["<1.0", "1.0–1.99", "2.0–2.99", "3.0–3.99", "4.0–5.0"])
    df["diagnosis_label"] = df.target.map({0: "No reported diagnosis", 1: "Reported diagnosis"})
    return df


def child_frame(demo, bmx):
    """Descriptive BMI-for-age categories for participants under 18 (no diabetes outcome is analysed)."""
    df = demo.merge(bmx, on="SEQN", how="left", validate="one_to_one")
    df = df[df.RIDAGEYR < 18].copy()
    df["child_age_band"] = pd.cut(df.RIDAGEYR, bins=[-1, 5, 11, 17], labels=["0–5 years", "6–11 years", "12–17 years"])
    df["bmi_category"] = df.BMDBMIC.map({1: "Underweight", 2: "Healthy weight", 3: "Overweight", 4: "Obesity"})
    return df


# ----------------------------------------------------------------------------- model
def build_pipeline():
    pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])
    return Pipeline([("prep", pre), ("model", LogisticRegression(max_iter=2500, class_weight="balanced"))])


def split_adults(adults):
    """Stratified 60/20/20 split with the same random states as the experiment (42, then 43)."""
    d = clean_predictors(adults)
    X, y = d[FEATURES], d["target"]
    Xtemp, Xte, ytemp, yte = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    Xtr, Xval, ytr, yval = train_test_split(Xtemp, ytemp, test_size=0.25, random_state=43, stratify=ytemp)
    return dict(Xtr=Xtr, ytr=ytr, Xval=Xval, yval=yval, Xte=Xte, yte=yte)


def train_final_model(adults):
    """Fit the final Logistic Regression pipeline on the TRAINING split only (as in the experiment)."""
    sp = split_adults(adults)
    pipe = build_pipeline().fit(sp["Xtr"], sp["ytr"])
    Xt = _dense(pipe.named_steps["prep"].transform(sp["Xtr"]))
    return dict(pipe=pipe, split=sp, train_mean=Xt.mean(axis=0),
                feature_names=list(pipe.named_steps["prep"].get_feature_names_out()))


def _dense(a):
    return a.toarray() if sparse.issparse(a) else np.asarray(a)


def classification_metrics(y, prob, thr=THRESHOLD):
    pred = (np.asarray(prob) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
                recall=recall_score(y, pred, zero_division=0), specificity=tn / (tn + fp),
                f1=f1_score(y, pred, zero_division=0), balanced_accuracy=balanced_accuracy_score(y, pred),
                roc_auc=roc_auc_score(y, prob), pr_auc=average_precision_score(y, prob),
                brier=brier_score_loss(y, prob), TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


def reproducibility_check(bundle):
    """Re-score the held-out test split with the live pipeline and compare with the saved CSV result."""
    sp = bundle["split"]
    prob = bundle["pipe"].predict_proba(sp["Xte"])[:, 1]
    live = classification_metrics(sp["yte"], prob)
    saved = pd.read_csv(RESULTS / "final_heldout_test_metrics.csv").iloc[0]
    rows = [("ROC-AUC", live["roc_auc"], saved.roc_auc), ("PR-AUC", live["pr_auc"], saved.pr_auc_average_precision),
            ("Accuracy", live["accuracy"], saved.accuracy), ("Recall", live["recall"], saved.recall_sensitivity),
            ("Specificity", live["specificity"], saved.specificity)]
    df = pd.DataFrame(rows, columns=["Metric", "Live pipeline in this app", "Saved experiment result"])
    df["Match"] = np.isclose(df.iloc[:, 1].astype(float), df.iloc[:, 2].astype(float), atol=1e-4)
    cm_match = [live["TN"], live["FP"], live["FN"], live["TP"]] == [int(saved.TN), int(saved.FP), int(saved.FN), int(saved.TP)]
    return df, bool(df["Match"].all() and cm_match), live


def predict_one(bundle, age, sex, race, edu, pir, bmi, sitting_min, threshold=THRESHOLD):
    """Run one hypothetical adult through the REAL trained pipeline (same cleaning as training)."""
    raw = pd.DataFrame([{"RIDAGEYR": float(age), "RIAGENDR": int(sex), "RIDRETH3": int(race), "DMDEDUC2": int(edu),
                         "INDFMPIR": float(pir), "BMXBMI": float(bmi), "PAD680": float(sitting_min)}])
    cleaned = clean_predictors(raw)[FEATURES]
    pipe = bundle["pipe"]
    score = float(pipe.predict_proba(cleaned)[0, 1])
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    xt = _dense(prep.transform(cleaned))[0]
    contrib = model.coef_[0] * (xt - bundle["train_mean"])  # linear-SHAP-style, relative to the average training profile
    rows = {}
    for name, c in zip(bundle["feature_names"], contrib):
        base = name.split("__", 1)[-1]
        for f in FEATURES:
            if base == f or base.startswith(f + "_"):
                rows[f] = rows.get(f, 0.0) + float(c)
                break
    contrib_df = pd.DataFrame({"feature": [FEATURE_LABELS[f] for f in rows], "contribution": list(rows.values())})
    contrib_df = contrib_df.sort_values("contribution")
    encoded = pd.DataFrame({"encoded feature": bundle["feature_names"], "value": xt})
    imputed = [FEATURE_LABELS[c] for c in FEATURES if pd.isna(cleaned.iloc[0][c])]
    return dict(score=score, positive=score >= threshold, threshold=threshold, contributions=contrib_df,
                encoded=encoded, raw=raw, cleaned=cleaned, imputed=imputed,
                intercept=float(model.intercept_[0]))


# ----------------------------------------------------------------------------- curves & saved results
def heldout_predictions():
    return pd.read_csv(RESULTS / "heldout_test_predictions.csv")


def roc_pr_curves(pred):
    fpr, tpr, _ = roc_curve(pred.y, pred.probability)
    prec, rec, _ = precision_recall_curve(pred.y, pred.probability)
    return dict(fpr=fpr, tpr=tpr, precision=prec, recall=rec, prevalence=float(pred.y.mean()),
                roc_auc=float(roc_auc_score(pred.y, pred.probability)),
                pr_auc=float(average_precision_score(pred.y, pred.probability)))


def missingness_table(adults):
    rows = []
    n = len(adults)
    special = {"RIDRETH3": [7, 9], "DMDEDUC2": [7, 9], "RIAGENDR": [7, 9], "PAD680": [7777, 9999]}
    for f in FEATURES:
        nan = int(adults[f].isna().sum())
        sp = int(adults[f].isin(special.get(f, [])).sum()) if f in special else 0
        rows.append({"Predictor": f"{FEATURE_LABELS[f]} ({f})", "Missing in raw file": nan,
                     "Special codes recoded to missing": sp, "Imputed in pipeline": nan + sp,
                     "% of 7,876 imputed": round(100 * (nan + sp) / n, 1),
                     "Imputation": "Median" if f in NUMERIC else "Most frequent"})
    return pd.DataFrame(rows)
