"""
Diabetes Across Ages - interactive ML dashboard (Streamlit).
Run locally with:  streamlit run app.py
"""
import inspect

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import core

# ----------------------------------------------------------------------------- theme
NAVY, DEEP, PRIMARY, MEDIUM, SOFT, PALE = "#082B4C", "#104A75", "#1769AA", "#3988C9", "#75B4E5", "#B9D9F2"
BG, CARD, TEXT, MUTED, AMBER = "#F4F8FC", "#FFFFFF", "#16324F", "#5B7390", "#E08A1E"
PALETTE = [PRIMARY, SOFT, DEEP, MEDIUM, PALE, NAVY]
UNWEIGHTED = "Unweighted analytic-sample proportions (NOT national prevalence estimates)."
DISCLAIMER = ("Academic model demonstration only. This is NOT a medical diagnosis, screening tool or clinical risk "
              "assessment. Do not use it to make health decisions.")

st.set_page_config(page_title="Diabetes Across Ages", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
.stApp {{ background: {BG}; color: {TEXT}; }}
.block-container {{ padding-top: 2rem; max-width: 1350px; }}
h1, h2, h3, h4 {{ color: {NAVY}; }}
section[data-testid="stSidebar"] {{ background: {NAVY}; }}
section[data-testid="stSidebar"] * {{ color: #E7F1FA; }}
section[data-testid="stSidebar"] hr {{ border-color: {DEEP}; }}
.kpi {{ background: {CARD}; border: 1px solid #DCE8F3; border-radius: 12px; padding: 16px 18px;
        box-shadow: 0 2px 8px rgba(8,43,76,.08); height: 100%; }}
.kpi-label {{ color: {MUTED}; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: .04em; }}
.kpi-value {{ color: {NAVY}; font-size: 30px; font-weight: 700; line-height: 1.25; }}
.kpi-sub {{ color: {MUTED}; font-size: 12.5px; }}
.card {{ background: {CARD}; border: 1px solid #DCE8F3; border-radius: 12px; padding: 18px 22px;
         box-shadow: 0 2px 8px rgba(8,43,76,.08); margin-bottom: 14px; }}
.card h4 {{ margin: 0 0 6px 0; color: {PRIMARY}; }}
.note {{ background: #EAF3FB; border: 1px solid {PALE}; border-radius: 10px; padding: 12px 16px; color: {TEXT}; font-size: 14.5px; }}
.warn {{ background: #FFF4E5; border: 1px solid #F3C98B; border-radius: 10px; padding: 12px 16px; color: #5A3A05; font-size: 14.5px; }}
.pill {{ display: inline-block; background: {PRIMARY}; color: #fff; border-radius: 999px; padding: 2px 12px; font-size: 12px; font-weight: 600; }}
.result-pos {{ background: {NAVY}; color: #fff; border-radius: 12px; padding: 18px 22px; }}
.result-neg {{ background: {DEEP}; color: #fff; border-radius: 12px; padding: 18px 22px; }}
.small {{ color: {MUTED}; font-size: 13px; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- helpers
def style(fig, height=420, legend_bottom=False):
    fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(color=TEXT, family="Segoe UI, Calibri, Arial, sans-serif"),
                      margin=dict(l=20, r=20, t=60, b=20), height=height, title_font_size=17)
    if legend_bottom:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))
    return fig


_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def show(fig, target=None):
    """Render a Plotly figure full-width on both older and newer Streamlit versions."""
    target = target if target is not None else st
    if _PLOTLY_HAS_WIDTH:
        target.plotly_chart(fig, width="stretch")
    else:
        target.plotly_chart(fig, use_container_width=True)


def kpi(col, label, value, sub=""):
    col.markdown(f"<div class='kpi'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div>"
                 f"<div class='kpi-sub'>{sub}</div></div>", unsafe_allow_html=True)


def card(title, body):
    st.markdown(f"<div class='card'><h4>{title}</h4>{body}</div>", unsafe_allow_html=True)


def note(text, kind="note"):
    st.markdown(f"<div class='{kind}'>{text}</div>", unsafe_allow_html=True)


def header(title, subtitle):
    st.markdown(f"## {title}")
    st.caption(subtitle)


def pct(v, d=2):
    return f"{v * 100:.{d}f}%"


# ----------------------------------------------------------------------------- cached data / model
@st.cache_data(show_spinner="Reading NHANES files...")
def get_sources():
    return core.load_sources()


@st.cache_data(show_spinner=False)
def get_adults():
    return core.adult_frame(get_sources()[4])


@st.cache_data(show_spinner=False)
def get_children():
    demo, _, bmx, _, _ = get_sources()
    return core.child_frame(demo, bmx)


@st.cache_resource(show_spinner="Training the Logistic Regression pipeline (training split only)...")
def get_model():
    return core.train_final_model(get_adults())


@st.cache_data(show_spinner=False)
def get_results():
    r = core.RESULTS
    return dict(
        final=pd.read_csv(r / "final_heldout_test_metrics.csv").iloc[0],
        age=pd.read_csv(r / "age_stratified_final_test_metrics.csv"),
        val=pd.read_csv(r / "validation_model_selection.csv"),
        sens=pd.read_csv(r / "validation_threshold_sensitivity.csv"),
        shap=pd.read_csv(r / "shap_global_importance.csv"),
        perm={"All adults": pd.read_csv(r / "permutation_importance_overall.csv"),
              core.AGE_GROUP_YOUNG: pd.read_csv(r / "permutation_importance_Adults_18_59.csv"),
              core.AGE_GROUP_OLD: pd.read_csv(r / "permutation_importance_Older_adults_60_plus.csv")},
        preds=core.heldout_predictions(),
    )


# ----------------------------------------------------------------------------- navigation
PAGES = ["Overview", "Diabetes Across Age Groups", "Machine Learning Explorer", "Model Performance",
         "Interactive ML Prediction", "Model Explainability", "Dataset & Methodology"]
if "page" not in st.session_state:
    st.session_state["page"] = PAGES[0]


def go_to(page):
    st.session_state["page"] = page


with st.sidebar:
    st.markdown("### 🩺 Diabetes Across Ages")
    st.caption("Age-stratified analytics and explainable ML | NHANES Aug 2021 – Aug 2023")
    st.divider()
    st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
    st.divider()
    st.markdown("**Scope**")
    st.caption("Classifier: adults aged 18+ only. Outcome: self-reported diagnosed diabetes. "
               "All results are unweighted and exploratory.")
    st.caption("Academic project. Not a medical tool.")

demo_df, diq_df, bmx_df, paq_df, merged_df = get_sources()
adults = get_adults()
R = get_results()
FINAL = R["final"]


# ----------------------------------------------------------------------------- 1. overview
def page_overview():
    header("Diabetes Across Ages", "B.Tech final-year major project | KIIT, School of Computer Science and Engineering")
    st.markdown("**How do diabetes-related patterns and the behaviour of a machine-learning classifier differ between "
                "adults aged 18–59 and older adults aged 60+?** This dashboard explores NHANES 2021–2023 public-use data, "
                "compares three classifiers, evaluates the selected Logistic Regression model on a held-out test set and "
                "explains it with SHAP.")
    c = st.columns(4)
    kpi(c[0], "NHANES participants", f"{len(demo_df):,}", "DEMO_L records merged on SEQN")
    kpi(c[1], "Eligible adults", f"{len(adults):,}", "Age 18+, DIQ010 answered Yes or No")
    kpi(c[2], "Reported diagnosis", f"{int(adults.target.sum()):,}", f"{int((1 - adults.target).sum()):,} without")
    kpi(c[3], "Sample proportion", pct(adults.target.mean()), "Unweighted, not national prevalence")
    st.write("")
    c = st.columns(4)
    kpi(c[0], "Selected model", "Logistic Regression", "Highest validation ROC-AUC")
    kpi(c[1], "Threshold", f"{core.THRESHOLD:.2f}", "Selected on validation data (max F1)")
    kpi(c[2], "Test ROC-AUC", f"{FINAL.roc_auc:.4f}", f"Held-out test, n = {int(FINAL.test_n):,}")
    kpi(c[3], "Test recall", pct(FINAL.recall_sensitivity), f"Precision {pct(FINAL.precision)}")
    st.write("")
    l, r = st.columns(2)
    young = adults[adults.age_group == core.AGE_GROUP_YOUNG].target.mean() * 100
    old = adults[adults.age_group == core.AGE_GROUP_OLD].target.mean() * 100
    with l:
        card("Research objective",
             "<ul><li>Compare diabetes-related patterns between adults 18–59 and older adults 60+.</li>"
             "<li>Examine BMI, demographic and lifestyle-related associations.</li>"
             "<li>Build and compare Logistic Regression, Random Forest and XGBoost classifiers.</li>"
             "<li>Evaluate on held-out data, by age group, and explain with SHAP.</li></ul>")
        card("Dataset", "NHANES August 2021 – August 2023 public-use files <b>DEMO_L, DIQ_L, BMX_L, PAQ_L</b>, merged one-to-one "
                        "on <b>SEQN</b>. Target: <b>DIQ010</b> (1 = Yes, 2 = No) among adults aged 18+. "
                        "Seven predictors: age, sex, race/ethnicity, education, income-to-poverty ratio, BMI, sedentary time.")
    with r:
        card("Key highlights",
             f"<ul><li>Unweighted sample proportion: <b>{young:.1f}%</b> (18–59) vs <b>{old:.1f}%</b> (60+).</li>"
             f"<li>Held-out test ROC-AUC <b>{FINAL.roc_auc:.3f}</b>; recall {pct(FINAL.recall_sensitivity, 1)}, "
             f"specificity {pct(FINAL.specificity, 1)}.</li>"
             "<li>Performance differs by age group (ROC-AUC 0.824 vs 0.663).</li>"
             "<li>Age is the most important model feature, then BMI.</li></ul>")
        note("<b>Scope:</b> the classifier is trained for adults aged 18 and above. There is <b>no</b> trained child diabetes "
             "classifier. All proportions are unweighted analytic-sample values, not national prevalence.")
    st.markdown("#### Explore the dashboard")
    cols = st.columns(3)
    for i, p in enumerate(PAGES[1:]):
        cols[i % 3].button(f"Open: {p}", key=f"nav_{i}", on_click=go_to, args=(p,))


# ----------------------------------------------------------------------------- 2. age groups
def page_age_groups():
    header("Diabetes Across Age Groups", "Adults 18–59 vs older adults 60+ | " + UNWEIGHTED)
    f1, f2, f3 = st.columns([1.4, 1.6, 1.2])
    grp = f1.radio("Age group", ["Both groups", core.AGE_GROUP_YOUNG, core.AGE_GROUP_OLD])
    lo, hi = f2.slider("Age range (80 = 80 and over)", 18, 80, (18, 80))
    sexes = f3.multiselect("Sex", ["Male", "Female"], default=["Male", "Female"])
    df = adults[(adults.RIDAGEYR >= lo) & (adults.RIDAGEYR <= hi) & (adults.sex_label.isin(sexes))]
    if grp != "Both groups":
        df = df[df.age_group == grp]
    if len(df) == 0:
        st.warning("No participants match the selected filters.")
        return
    c = st.columns(4)
    kpi(c[0], "Participants", f"{len(df):,}", "Matching the filters")
    kpi(c[1], "Reported diagnosis", f"{int(df.target.sum()):,}", "DIQ010 = Yes")
    kpi(c[2], "Sample proportion", pct(df.target.mean()), "Unweighted")
    kpi(c[3], "Median BMI", f"{df.BMXBMI.median():.1f}" if df.BMXBMI.notna().any() else "n/a", "kg/m², where measured")
    st.write("")
    t1, t2, t3, t4 = st.tabs(["Age-wise pattern", "Case distribution", "BMI comparison", "Demographic & lifestyle"])

    with t1:
        g = df.groupby("age_band", observed=True).target.agg(["size", "sum", "mean"]).reset_index()
        g["Sample proportion (%)"] = 100 * g["mean"]
        fig = px.bar(g, x="age_band", y="Sample proportion (%)", text=g["Sample proportion (%)"].round(1).astype(str) + "%",
                     hover_data={"size": True, "sum": True, "mean": False},
                     labels={"age_band": "Age band (years)", "size": "Participants", "sum": "Reported cases"},
                     title="Reported diagnosis by age band", color_discrete_sequence=[PRIMARY])
        fig.update_traces(textposition="outside")
        show(style(fig))
        st.caption(UNWEIGHTED + " Bands with few participants give unstable percentages; hover for counts.")
        fig = px.histogram(df, x="RIDAGEYR", color="diagnosis_label", nbins=32, barmode="overlay", opacity=0.75,
                           color_discrete_sequence=[SOFT, DEEP], title="Age distribution by reported diagnosis",
                           labels={"RIDAGEYR": "Age (years)", "diagnosis_label": ""})
        show(style(fig, 380, True))

    with t2:
        a, b = st.columns(2)
        gg = df.groupby(["age_group", "diagnosis_label"]).size().reset_index(name="participants")
        fig = px.bar(gg, x="age_group", y="participants", color="diagnosis_label", barmode="stack", text="participants",
                     color_discrete_sequence=[SOFT, DEEP], title="Participants by age group and reported diagnosis",
                     labels={"age_group": "", "participants": "Participants", "diagnosis_label": ""})
        show(style(fig, 400, True), a)
        cases = df[df.target == 1].groupby("age_group").size().reset_index(name="cases")
        fig = px.pie(cases, names="age_group", values="cases", hole=0.5, color_discrete_sequence=[SOFT, DEEP],
                     title="Where the reported cases fall (share of cases)")
        show(style(fig, 400, True), b)
        sm = df.groupby("age_group").target.agg(participants="size", cases="sum", proportion="mean").reset_index()
        sm["sample proportion"] = (sm.proportion * 100).round(2).astype(str) + "%"
        st.dataframe(sm[["age_group", "participants", "cases", "sample proportion"]], hide_index=True)
        st.caption(UNWEIGHTED)

    with t3:
        a, b = st.columns(2)
        d2 = df.dropna(subset=["BMXBMI"])
        fig = px.box(d2, x="age_group", y="BMXBMI", color="diagnosis_label", color_discrete_sequence=[SOFT, DEEP],
                     title="BMI by age group and reported diagnosis",
                     labels={"age_group": "", "BMXBMI": "BMI (kg/m²)", "diagnosis_label": ""})
        show(style(fig, 420, True), a)
        bc = d2.groupby(["age_group", "bmi_category"], observed=True).target.agg(["size", "mean"]).reset_index()
        bc["Sample proportion (%)"] = 100 * bc["mean"]
        fig = px.bar(bc, x="bmi_category", y="Sample proportion (%)", color="age_group", barmode="group",
                     color_discrete_sequence=[SOFT, DEEP], hover_data={"size": True, "mean": False},
                     title="Reported diagnosis by BMI category",
                     labels={"bmi_category": "BMI category", "age_group": "", "size": "Participants"})
        show(style(fig, 420, True), b)
        st.caption(UNWEIGHTED + f" BMI is missing for {int(df.BMXBMI.isna().sum()):,} of the {len(df):,} filtered "
                                "participants and they are excluded from these two charts. Associations only, not causation.")

    with t4:
        opts = {"Sex": "sex_label", "Race/ethnicity": "race_label", "Education": "edu_label",
                "Daily sitting time": "sitting_band", "Income-to-poverty ratio": "pir_band"}
        choice = st.selectbox("Factor to compare across age groups", list(opts))
        col = opts[choice]
        d3 = df.dropna(subset=[col])
        gg = d3.groupby([col, "age_group"], observed=True).target.agg(["size", "mean"]).reset_index()
        gg["Sample proportion (%)"] = 100 * gg["mean"]
        fig = px.bar(gg, x=col, y="Sample proportion (%)", color="age_group", barmode="group",
                     color_discrete_sequence=[SOFT, DEEP], hover_data={"size": True, "mean": False},
                     title=f"Reported diagnosis by {choice.lower()}", labels={col: choice, "age_group": "", "size": "Participants"})
        show(style(fig, 440, True))
        st.caption(UNWEIGHTED + " Small groups give unstable percentages; hover to see participant counts.")
        with st.expander("Children (under 18): descriptive BMI-for-age only"):
            ch = get_children().dropna(subset=["bmi_category"])
            g = ch.groupby(["child_age_band", "bmi_category"], observed=True).size().reset_index(name="participants")
            fig = px.bar(g, x="child_age_band", y="participants", color="bmi_category", barmode="stack",
                         color_discrete_sequence=PALETTE, title="Child BMI-for-age categories (non-missing records)",
                         labels={"child_age_band": "Age band", "bmi_category": "BMI-for-age"})
            show(style(fig, 380, True))
            st.caption("No child diabetes outcome or classifier is analysed in this project; this is a descriptive BMI-for-age profile only.")


# ----------------------------------------------------------------------------- 3. ML explorer
def page_ml_explorer():
    header("Machine Learning Explorer", "Model development, comparison and selection | validation results")
    st.markdown("#### Model development process")
    steps = [("1", "Merge & filter", "Adults 18+ with DIQ010 = Yes/No (n = 7,876)"),
             ("2", "Split", "Stratified 60/20/20: 4,725 train, 1,575 validation, 1,576 test"),
             ("3", "Preprocess", "Impute + scale numeric; impute + one-hot categorical (fit on train)"),
             ("4", "Train 3 models", "Logistic Regression, Random Forest, XGBoost with class weighting"),
             ("5", "Select on validation", "Best validation ROC-AUC; threshold = max validation F1"),
             ("6", "Test once", "Final held-out test evaluation of the selected model")]
    cols = st.columns(6)
    for col, (n, t, d) in zip(cols, steps):
        kpi(col, f"Step {n}", f"<span style='font-size:18px'>{t}</span>", d)
    st.write("")
    a, b, c = st.columns(3)
    with a:
        card("Logistic Regression", "Linear model, <code>class_weight='balanced'</code>, max_iter 2500. Transparent: each feature has a coefficient. "
                                    "<span class='pill'>SELECTED</span>")
    with b:
        card("Random Forest", "450 trees, min leaf size 5, <code>max_features='sqrt'</code>, <code>class_weight='balanced_subsample'</code>.")
    with c:
        card("XGBoost", "450 boosted trees, depth 3, learning rate 0.035, subsample 0.85, L2 = 2.0, <code>scale_pos_weight</code> ≈ 6.34.")
    val = R["val"].copy()
    st.markdown("#### Model comparison (validation set, n = 1,575)")
    vs = val.sort_values("validation_roc_auc")
    fig = px.bar(vs, x="validation_roc_auc", y="model", orientation="h", text=vs.validation_roc_auc.round(4),
                 color_discrete_sequence=[PRIMARY], title="Validation ROC-AUC",
                 labels={"validation_roc_auc": "Validation ROC-AUC", "model": ""})
    fig.update_traces(textposition="outside")
    fig.update_xaxes(range=[0.7, 0.82])
    show(style(fig, 300))
    tbl = val[["model", "validation_threshold", "validation_roc_auc", "validation_pr_auc_average_precision",
                "validation_accuracy", "validation_precision", "validation_recall_sensitivity", "validation_specificity",
                "validation_f1", "validation_balanced_accuracy"]].copy()
    tbl.columns = ["Model", "Threshold", "ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "Specificity", "F1", "Balanced acc."]
    st.dataframe(tbl.round(4), hide_index=True)
    st.markdown("#### Threshold sensitivity (validation set)")
    sens = R["sens"]
    models = list(sens.model.unique())
    mdl = st.selectbox("Model", models, index=models.index("Logistic Regression"))
    s = sens[sens.model == mdl].sort_values("threshold")
    long = s.melt(id_vars="threshold", value_vars=["precision", "recall_sensitivity", "specificity", "f1"],
                  var_name="metric", value_name="value")
    long["metric"] = long.metric.map({"precision": "Precision", "recall_sensitivity": "Recall", "specificity": "Specificity", "f1": "F1"})
    fig = px.line(long, x="threshold", y="value", color="metric", markers=True, color_discrete_sequence=PALETTE,
                  title=f"{mdl}: metrics at evaluated thresholds", labels={"threshold": "Classification threshold", "value": "Metric value", "metric": ""})
    show(style(fig, 380, True))
    st.caption("Only the thresholds saved by the experiment (0.20, 0.30, 0.40, 0.50 and each model's validation-optimal value) are shown. "
               "Raising the threshold trades recall for precision and specificity.")
    st.markdown("#### Why Logistic Regression was selected")
    note("<b>Selection methodology:</b> models were ranked by <b>validation ROC-AUC</b>, a threshold-independent measure, and "
         "Logistic Regression ranked first (0.7944 vs 0.7904 for XGBoost and 0.7816 for Random Forest). Its classification "
         f"threshold, <b>{core.THRESHOLD:.2f}</b>, maximised F1 on the validation set. The test set was used once, after selection. "
         "The AUC differences between models are small, and the linear model is also the simplest and most interpretable, "
         "so selecting it is well supported but not a large margin.")


# ----------------------------------------------------------------------------- 4. performance
def confusion_fig(tn, fp, fn, tp, title):
    z = [[tn, fp], [fn, tp]]
    fig = px.imshow(z, x=["Predicted negative", "Predicted positive"], y=["Actual negative", "Actual positive"],
                    text_auto=True, color_continuous_scale=[[0, "#F4F8FC"], [1, DEEP]], title=title,
                    labels=dict(color="Count"))
    fig.update_coloraxes(showscale=False)
    fig.update_traces(textfont_size=22)
    return style(fig, 380)


def metric_cards(m):
    c = st.columns(5)
    kpi(c[0], "Accuracy", pct(m["accuracy"]), "Overall correct")
    kpi(c[1], "Precision", pct(m["precision"]), "Predicted positives that are real")
    kpi(c[2], "Recall", pct(m["recall"]), "Reported cases found")
    kpi(c[3], "Specificity", pct(m["specificity"]), "Non-cases correctly negative")
    kpi(c[4], "F1-score", pct(m["f1"]), "Precision/recall balance")
    st.write("")
    c = st.columns(4)
    kpi(c[0], "Balanced accuracy", pct(m["balanced_accuracy"]), "Mean of recall & specificity")
    kpi(c[1], "ROC-AUC", f"{m['roc_auc']:.4f}", "Ranking ability")
    kpi(c[2], "PR-AUC", f"{m['pr_auc']:.4f}", "Average precision")
    kpi(c[3], "Brier score", f"{m['brier']:.4f}", "Lower is better")


def page_performance():
    header("Model Performance", "Logistic Regression, threshold 0.64 | validation and held-out test results are shown separately")
    t1, t2, t3 = st.tabs(["Held-out test results", "Validation results", "Test results by age group"])
    with t1:
        st.markdown(f"<span class='pill'>HELD-OUT TEST</span> &nbsp; n = {int(FINAL.test_n):,}, positives = {int(FINAL.test_positive_n)} "
                    f"({pct(FINAL.test_prevalence)}), used once after model and threshold selection.", unsafe_allow_html=True)
        st.write("")
        m = dict(accuracy=FINAL.accuracy, precision=FINAL.precision, recall=FINAL.recall_sensitivity, specificity=FINAL.specificity,
                 f1=FINAL.f1, balanced_accuracy=FINAL.balanced_accuracy, roc_auc=FINAL.roc_auc, pr_auc=FINAL.pr_auc_average_precision,
                 brier=FINAL.brier_score)
        metric_cards(m)
        st.write("")
        a, b = st.columns(2)
        show(confusion_fig(int(FINAL.TN), int(FINAL.FP), int(FINAL.FN), int(FINAL.TP), "Confusion matrix (test)"), a)
        cur = core.roc_pr_curves(R["preds"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cur["fpr"], y=cur["tpr"], mode="lines", name=f"Logistic Regression (AUC {cur['roc_auc']:.3f})", line=dict(color=PRIMARY, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color=MUTED, dash="dash")))
        fig.add_trace(go.Scatter(x=[1 - FINAL.specificity], y=[FINAL.recall_sensitivity], mode="markers", name="Operating point (0.64)",
                                 marker=dict(color=AMBER, size=12, line=dict(color="white", width=1))))
        fig.update_layout(title="ROC curve (test)", xaxis_title="False positive rate", yaxis_title="True positive rate")
        show(style(fig, 380, True), b)
        a, b = st.columns(2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cur["recall"], y=cur["precision"], mode="lines", name=f"Logistic Regression (AP {cur['pr_auc']:.3f})", line=dict(color=PRIMARY, width=3)))
        fig.add_hline(y=cur["prevalence"], line_dash="dash", line_color=MUTED, annotation_text=f"No-skill = prevalence {cur['prevalence']:.3f}")
        fig.add_trace(go.Scatter(x=[FINAL.recall_sensitivity], y=[FINAL.precision], mode="markers", name="Operating point (0.64)",
                                 marker=dict(color=AMBER, size=12, line=dict(color="white", width=1))))
        fig.update_layout(title="Precision-recall curve (test)", xaxis_title="Recall", yaxis_title="Precision")
        show(style(fig, 380, True), a)
        with b:
            card("How to read this",
                 "<ul><li><b>Recall 53.95%:</b> about half of the reported cases are found.</li>"
                 "<li><b>Precision 32.31%:</b> roughly one in three flagged people is a reported case; the proportion in the sample is only 13.6%.</li>"
                 "<li><b>Accuracy</b> alone is not enough with imbalanced classes.</li>"
                 "<li><b>ROC-AUC 0.788:</b> reasonable ranking ability, modest classification quality at this threshold.</li></ul>")
        note("These are exploratory, unweighted results from a single survey cycle with no external validation.", "warn")
    with t2:
        v = R["val"].set_index("model").loc["Logistic Regression"]
        st.markdown("<span class='pill'>VALIDATION</span> &nbsp; n = 1,575. Used to choose the model and the threshold, so these numbers are "
                    "optimistic compared with the untouched test set.", unsafe_allow_html=True)
        st.write("")
        m = dict(accuracy=v.validation_accuracy, precision=v.validation_precision, recall=v.validation_recall_sensitivity,
                 specificity=v.validation_specificity, f1=v.validation_f1, balanced_accuracy=v.validation_balanced_accuracy,
                 roc_auc=v.validation_roc_auc, pr_auc=v.validation_pr_auc_average_precision, brier=v.validation_brier_score)
        metric_cards(m)
        st.write("")
        a, b = st.columns(2)
        show(confusion_fig(int(v.validation_TN), int(v.validation_FP), int(v.validation_FN), int(v.validation_TP), "Confusion matrix (validation)"), a)
        cmp = pd.DataFrame({"Metric": ["ROC-AUC", "PR-AUC", "Recall", "Specificity", "F1"],
                            "Validation": [v.validation_roc_auc, v.validation_pr_auc_average_precision, v.validation_recall_sensitivity, v.validation_specificity, v.validation_f1],
                            "Held-out test": [FINAL.roc_auc, FINAL.pr_auc_average_precision, FINAL.recall_sensitivity, FINAL.specificity, FINAL.f1]})
        long = cmp.melt(id_vars="Metric", var_name="Set", value_name="Value")
        fig = px.bar(long, x="Metric", y="Value", color="Set", barmode="group", color_discrete_sequence=[SOFT, DEEP],
                     title="Validation vs held-out test", text=long.Value.round(3))
        show(style(fig, 380, True), b)
        st.caption("ROC and PR curves are shown for the test set only, because row-level validation predictions were not saved by the experiment.")
    with t3:
        age = R["age"]
        st.markdown("<span class='pill'>HELD-OUT TEST, SUBGROUPS</span> &nbsp; Same model and same fixed threshold (0.64). Descriptive only.", unsafe_allow_html=True)
        st.write("")
        cols = st.columns(2)
        for col, (_, r) in zip(cols, age.iterrows()):
            with col:
                card(r.age_group,
                     f"Test samples <b>{int(r.n_test)}</b> | cases <b>{int(r.positive_n)}</b> ({pct(r.unweighted_prevalence)})<br>"
                     f"Recall <b>{pct(r.recall_sensitivity)}</b> | specificity <b>{pct(r.specificity)}</b> | precision <b>{pct(r.precision)}</b><br>"
                     f"ROC-AUC <b>{r.roc_auc:.3f}</b> | PR-AUC <b>{r.pr_auc_average_precision:.3f}</b>")
        long = age.melt(id_vars="age_group", value_vars=["recall_sensitivity", "specificity", "precision", "roc_auc"], var_name="metric", value_name="value")
        long["metric"] = long.metric.map({"recall_sensitivity": "Recall", "specificity": "Specificity", "precision": "Precision", "roc_auc": "ROC-AUC"})
        fig = px.bar(long, x="metric", y="value", color="age_group", barmode="group", color_discrete_sequence=[SOFT, DEEP],
                     text=long.value.round(3), title="Held-out test metrics by age group", labels={"metric": "", "value": "Value", "age_group": ""})
        show(style(fig, 400, True))
        a, b = st.columns(2)
        for col, (_, r) in zip([a, b], age.iterrows()):
            show(confusion_fig(int(r.TN), int(r.FP), int(r.FN), int(r.TP), f"Confusion matrix: {r.age_group}"), col)
        note("<b>Reading the difference:</b> with one fixed threshold, the model is conservative in adults 18–59 (high specificity, low recall) and "
             "more liberal in adults 60+ (higher recall, lower specificity). The sample proportion differs (7.22% vs 22.19%), and age, the "
             "dominant feature, separates cases much better among younger adults (permutation importance of age 0.153 vs 0.008). The 18–59 "
             "subgroup has only 65 test cases, so its estimates are uncertain.")


# ----------------------------------------------------------------------------- 5. prediction
def page_prediction():
    header("Interactive ML Prediction", "Run a hypothetical adult profile through the trained Logistic Regression pipeline")
    note(DISCLAIMER, "warn")
    st.write("")
    st.write("The model estimates a score for matching the **self-reported diagnosed-diabetes label** in the NHANES adult sample. "
             "It does not estimate future disease risk.")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Age (years; NHANES records 80 for ages 80+)", 18, 80, 45)
            sex = st.radio("Sex", list(core.SEX_LABELS), format_func=core.SEX_LABELS.get, horizontal=True)
            race = st.selectbox("Race/ethnicity", list(core.RACE_LABELS), index=2, format_func=core.RACE_LABELS.get)
        with c2:
            edu = st.selectbox("Education level", list(core.EDU_LABELS), index=4, format_func=core.EDU_LABELS.get)
            pir = st.number_input("Family income-to-poverty ratio (0 to 5)", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
            bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=75.0, value=27.0, step=0.1)
        with c3:
            sit = st.slider("Sedentary time (minutes per day)", 0, 1440, 360, 15)
            st.caption(f"Sedentary time in the form above is NHANES PAD680 (minutes per day).")
        submitted = st.form_submit_button("Run model prediction", type="primary")
    st.caption("Notes: NHANES records adult education only for ages 20+, so in the training data ages 18–19 had education imputed. "
               "The trained pipeline also recodes the race/ethnicity category 'Other race, including multi-racial' as missing and imputes the "
               "most frequent category, exactly as in the saved experiment.")
    if not submitted:
        return
    bundle = get_model()
    out = core.predict_one(bundle, age, sex, race, edu, pir, bmi, sit)
    st.markdown("### Model output")
    a, b = st.columns([1, 1.2])
    with a:
        cls = "result-pos" if out["positive"] else "result-neg"
        label = "POSITIVE class" if out["positive"] else "NEGATIVE class"
        st.markdown(f"<div class='{cls}'><div style='font-size:13px;opacity:.8'>MODEL CLASSIFICATION (threshold {out['threshold']:.2f})</div>"
                    f"<div style='font-size:30px;font-weight:700'>{label}</div>"
                    f"<div style='font-size:15px'>Model score: <b>{out['score']:.3f}</b> ({out['score'] * 100:.1f}%)</div></div>", unsafe_allow_html=True)
        st.write("")
        if out["positive"]:
            st.write(f"**In simple terms:** the model's score for this profile ({out['score']:.2f}) is at or above the cut-off ({out['threshold']:.2f}), "
                     "so it labels this profile as resembling NHANES adults who reported a diabetes diagnosis.")
        else:
            st.write(f"**In simple terms:** the model's score for this profile ({out['score']:.2f}) is below the cut-off ({out['threshold']:.2f}), "
                     "so it labels this profile as resembling NHANES adults who did not report a diagnosis.")
        st.caption("The score comes from a class-weighted model, so it is not a calibrated real-world probability or personal risk.")
    with b:
        fig = go.Figure(go.Indicator(mode="gauge+number", value=out["score"] * 100, number={"suffix": "%"},
                                     title={"text": "Model score"},
                                     gauge={"axis": {"range": [0, 100]}, "bar": {"color": PRIMARY},
                                            "steps": [{"range": [0, out["threshold"] * 100], "color": "#EAF3FB"},
                                                      {"range": [out["threshold"] * 100, 100], "color": PALE}],
                                            "threshold": {"line": {"color": AMBER, "width": 4}, "thickness": 0.8, "value": out["threshold"] * 100}}))
        show(style(fig, 280))
        st.caption("Amber line = classification threshold (0.64).")
    if out["imputed"]:
        st.info("Treated as missing and imputed by the trained pipeline: " + ", ".join(out["imputed"]) + ".")
    st.markdown("#### What pushed this score up or down?")
    cd = out["contributions"]
    cd = cd.assign(direction=np.where(cd.contribution >= 0, "Raises score", "Lowers score"))
    fig = px.bar(cd, x="contribution", y="feature", orientation="h", color="direction",
                 color_discrete_map={"Raises score": DEEP, "Lowers score": SOFT},
                 title="Contribution to the model's log-odds relative to the average training profile",
                 labels={"contribution": "Contribution (log-odds)", "feature": "", "direction": ""})
    show(style(fig, 360, True))
    st.caption("Linear-SHAP-style contributions computed from this model's coefficients. They describe how the model combines inputs, not causes.")
    with st.expander("Show exactly what the pipeline received (input encoding)"):
        st.markdown("**Input row** (after the same recoding used in training):")
        st.dataframe(out["cleaned"], hide_index=True)
        st.markdown("**Model-ready vector** (numeric columns standardised with training mean/SD; categories one-hot encoded):")
        st.dataframe(out["encoded"].round(4), hide_index=True)
        st.caption(f"Logistic Regression intercept: {out['intercept']:.4f}. Score = sigmoid(intercept + coefficients x vector).")
    note(DISCLAIMER, "warn")


# ----------------------------------------------------------------------------- 6. explainability
def page_explainability():
    header("Model Explainability", "SHAP and permutation importance for the Logistic Regression model")
    shap_df = R["shap"].copy()
    shap_df["Feature"] = shap_df.feature.map(core.FEATURE_LABELS)
    sd = shap_df.sort_values("mean_absolute_shap_value")
    a, b = st.columns(2)
    fig = px.bar(sd, x="mean_absolute_shap_value", y="Feature", orientation="h",
                 text=sd.mean_absolute_shap_value.round(3), color_discrete_sequence=[PRIMARY],
                 title="Global SHAP importance (mean |SHAP value|)",
                 labels={"mean_absolute_shap_value": "Mean absolute SHAP value"})
    fig.update_traces(textposition="outside")
    show(style(fig, 440), a)
    which = b.selectbox("Permutation importance subset (held-out test, ROC-AUC drop)", list(R["perm"]))
    pm = R["perm"][which].copy()
    pm["Feature"] = pm.feature.map(core.FEATURE_LABELS)
    pm = pm.sort_values("importance_mean")
    fig = px.bar(pm, x="importance_mean", y="Feature", orientation="h", error_x="importance_std",
                 color_discrete_sequence=[DEEP], title=f"Permutation importance: {which}",
                 labels={"importance_mean": "Mean drop in ROC-AUC when shuffled"})
    show(style(fig, 380), b)
    note("SHAP shows how much each input moves the model's output on average; permutation importance shows how much test ROC-AUC "
         "falls when a feature is shuffled. Both agree that <b>age</b> and <b>BMI</b> matter most overall. Within older adults, age is "
         "much less important (permutation 0.008), probably because that group covers a narrower age range, and BMI leads. "
         "<b>Feature importance describes the model, not causes: it does not show that any feature causes diabetes.</b>")
    st.markdown("#### What the features mean")
    gl = pd.DataFrame([
        ["Age", "RIDAGEYR", "Age in years at screening (80 = 80 and over).", "Strongest signal: reported diagnosis becomes more common with age."],
        ["BMI", "BMXBMI", "Measured body-mass index (kg/m²).", "Higher BMI is associated with a higher score."],
        ["Education", "DMDEDUC2", "Highest education level (adults 20+).", "A socioeconomic marker; lower education is associated with higher scores in this sample."],
        ["Race/ethnicity", "RIDRETH3", "NHANES race/Hispanic-origin categories.", "Captures group differences in the sample; not a biological cause."],
        ["Sex", "RIAGENDR", "Male or female.", "Smaller contribution."],
        ["Sedentary time", "PAD680", "Self-reported minutes sitting per day.", "Smaller contribution; self-reported."],
        ["Income-to-poverty ratio", "INDFMPIR", "Family income relative to the poverty line (0 to 5).", "Smallest contribution; a socioeconomic marker."],
    ], columns=["Feature", "NHANES variable", "Meaning", "Role in the model (association only)"])
    st.dataframe(gl, hide_index=True)
    st.caption("Direction comments describe associations learned from this unweighted sample, not causal effects.")
    with st.expander("SHAP method"):
        st.write("Linear SHAP on the transformed design matrix of the Logistic Regression model; one-hot columns are summed back to their "
                 "source variable. Values come from results/shap_global_importance.csv produced by add_shap.py.")


# ----------------------------------------------------------------------------- 7. methodology
def page_methodology():
    header("Dataset & Methodology", "How the data were assembled, processed, modelled and evaluated")
    tabs = st.tabs(["Dataset", "Merging & target", "Features & preprocessing", "Split & training", "Evaluation", "Limitations", "Reproducibility check"])
    with tabs[0]:
        st.markdown("**NHANES** (National Health and Nutrition Examination Survey) is a US national health survey. "
                    "This project uses the public-use files for **August 2021 – August 2023**.")
        files = pd.DataFrame([
            ["DEMO_L", "Demographics", len(demo_df), "Age, sex, race/ethnicity, education, income-to-poverty ratio"],
            ["DIQ_L", "Diabetes questionnaire", len(diq_df), "DIQ010: doctor-told diabetes (target)"],
            ["BMX_L", "Body measures", len(bmx_df), "BMXBMI (and child BMI-for-age category)"],
            ["PAQ_L", "Physical activity", len(paq_df), "PAD680: minutes of sedentary activity"],
        ], columns=["File", "Content", "Records", "Variables used"])
        st.dataframe(files, hide_index=True)
    with tabs[1]:
        n_adult = int((merged_df.RIDAGEYR >= 18).sum())
        c = st.columns(4)
        kpi(c[0], "Merged participants", f"{len(merged_df):,}", "One row per SEQN")
        kpi(c[1], "Adults 18+", f"{n_adult:,}", "")
        kpi(c[2], "Eligible", f"{len(adults):,}", "DIQ010 = 1 or 2")
        kpi(c[3], "Excluded adults", f"{n_adult - len(adults):,}", "Borderline, refused, don't know, missing")
        st.write("")
        st.markdown("- The four files are merged one-to-one on **SEQN** (validated at merge time).\n"
                    "- **Target:** self-reported diagnosed diabetes, DIQ010 = 1 (Yes) coded 1 and DIQ010 = 2 (No) coded 0, adults 18+.\n"
                    f"- **Positive cases:** {int(adults.target.sum()):,}; **negative:** {int((1 - adults.target).sum()):,}; "
                    f"**unweighted positive proportion:** {pct(adults.target.mean())}.\n"
                    "- Excluded from predictors: DIQ010 itself, **HbA1c** (a diagnostic measure, so it would leak the outcome), SEQN and survey weights.")
    with tabs[2]:
        st.markdown("**Predictors:** age (RIDAGEYR), sex (RIAGENDR), race/ethnicity (RIDRETH3), education (DMDEDUC2), "
                    "income-to-poverty ratio (INDFMPIR), BMI (BMXBMI), sedentary time (PAD680).")
        st.markdown("**Missing-value handling:** special codes (7 and 9 for categorical variables; 7777 and 9999 for sedentary time) are recoded to missing. "
                    "Numeric variables are median-imputed and standardised; categorical variables are most-frequent-imputed and one-hot encoded. "
                    "All of this is fitted inside the training pipeline only.")
        st.dataframe(core.missingness_table(adults), hide_index=True)
        note("Race/ethnicity code 7 ('Other race, including multi-racial') is a valid NHANES category, but the experiment's recoding step treats it as missing, "
             "so those 529 participants are imputed with the most frequent category. The dashboard reproduces this exactly so that it matches the saved results.", "warn")
    with tabs[3]:
        c = st.columns(3)
        kpi(c[0], "Train", "4,725", "60% (stratified)")
        kpi(c[1], "Validation", "1,575", "20%: model and threshold selection")
        kpi(c[2], "Test", "1,576", "20%: used once at the end")
        st.write("")
        st.markdown("- Two stratified `train_test_split` calls (random states 42 and 43) keep the class proportion in every split.\n"
                    "- Three models are trained on the training split with class weighting for the minority class.\n"
                    "- The model with the highest **validation ROC-AUC** is chosen and its threshold is set by maximum **validation F1**.")
    with tabs[4]:
        st.markdown("- Metrics: accuracy, precision, recall (sensitivity), specificity, F1, balanced accuracy, ROC-AUC, PR-AUC (average precision), Brier score, confusion matrix.\n"
                    "- Validation results guide model choice; the held-out test set provides the final estimate.\n"
                    "- Test results are also reported separately for adults 18–59 and 60+ (descriptive).\n"
                    "- Explainability: linear SHAP and permutation importance.")
    with tabs[5]:
        for t in ["Modelling is **unweighted**; the NHANES complex survey design was not incorporated. Results are not national prevalence estimates.",
                  "One survey cycle (2021–2023) and **no external validation**.",
                  "Diabetes status is **self-reported**; undiagnosed diabetes may not be captured.",
                  "The classifier is for **adults only**; there is no child classifier.",
                  "Feature importance **does not establish causation**.",
                  "The 0.64 threshold maximises validation F1 and is **not a medically recommended** decision threshold.",
                  "Race/ethnicity code 7 is recoded as missing in the preprocessing step (see Features & preprocessing).",
                  "This application is an academic demonstration, **not** a diagnostic or clinical risk tool."]:
            st.markdown(f"- {t}")
    with tabs[6]:
        st.write("Re-train the Logistic Regression pipeline from the bundled NHANES files (training split only) and compare its test-set results with the saved experiment results.")
        if st.button("Run reproducibility check"):
            table, ok, _ = core.reproducibility_check(get_model())
            st.dataframe(table.round(6), hide_index=True)
            if ok:
                st.success("The live pipeline reproduces the saved held-out test results, including the confusion matrix.")
            else:
                st.error("The live pipeline does not exactly match the saved results (this can happen with a different scikit-learn version).")


# ----------------------------------------------------------------------------- router
ROUTES = {"Overview": page_overview, "Diabetes Across Age Groups": page_age_groups,
          "Machine Learning Explorer": page_ml_explorer, "Model Performance": page_performance,
          "Interactive ML Prediction": page_prediction, "Model Explainability": page_explainability,
          "Dataset & Methodology": page_methodology}
ROUTES[st.session_state["page"]]()
