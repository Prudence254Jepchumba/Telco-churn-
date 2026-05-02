"""
Telco Customer Churn Analysis Dashboard
A professional Streamlit application for end-to-end churn prediction analysis.
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import io
import base64

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import shap
import lime
import lime.lime_tabular

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global styling ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E2761;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #1E2761;
    }
    .best-model-badge {
        background: #1E2761;
        color: white;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    div[data-testid="stTabs"] button {
        font-size: 0.95rem;
        font-weight: 500;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1E2761;
        border-bottom: 2px solid #e9ecef;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Helper: convert matplotlib figure to PNG bytes for download ───────────────
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

# ── Helper: convert dataframe to CSV bytes ────────────────────────────────────
def df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING AND PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))

    # Fix TotalCharges type and impute missing values
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop identifier column
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)

    # Impute ALL numeric NaNs with median (covers TotalCharges + any other numeric cols)
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Impute ALL categorical NaNs with mode
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


@st.cache_data(show_spinner=False)
def preprocess(df):
    df_enc = df.copy()
    df_enc["Churn"] = df_enc["Churn"].map({"Yes": 1, "No": 0})

    # One-hot encode all remaining categorical columns
    cat_cols = df_enc.select_dtypes(include="object").columns.tolist()
    df_enc = pd.get_dummies(df_enc, columns=cat_cols, drop_first=True)

    X = df_enc.drop("Churn", axis=1)
    y = df_enc["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply SMOTE only to training data to address class imbalance
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Scale features — fit on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled  = scaler.transform(X_test)

    feature_names = X.columns.tolist()

    return X_train_scaled, y_train_res, X_test_scaled, y_test, feature_names, X_test


@st.cache_data(show_spinner=False)
def train_all_models(_X_train, _y_train, _X_test, _y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "XGBoost":             XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                                              eval_metric="logloss", random_state=42, verbosity=0),
        "LightGBM":            LGBMClassifier(n_estimators=200, learning_rate=0.05,
                                               random_state=42, verbose=-1),
        "CatBoost":            CatBoostClassifier(iterations=200, learning_rate=0.05,
                                                   depth=6, random_state=42, verbose=0),
    }

    results = {}
    trained = {}

    for name, model in models.items():
        model.fit(_X_train, _y_train)
        preds = model.predict(_X_test)
        probs = model.predict_proba(_X_test)[:, 1]

        results[name] = {
            "Accuracy":  round(accuracy_score(_y_test, preds), 4),
            "Precision": round(precision_score(_y_test, preds), 4),
            "Recall":    round(recall_score(_y_test, preds), 4),
            "F1-Score":  round(f1_score(_y_test, preds), 4),
            "AUC-ROC":   round(roc_auc_score(_y_test, probs), 4),
            "MCC":       round(matthews_corrcoef(_y_test, preds), 4),
            "preds":     preds,
            "probs":     probs,
        }
        trained[name] = model

    return results, trained


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## Telco Churn Dashboard")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload the IBM Telco Churn CSV file",
        type=["csv"],
        help="Upload WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    if st.button("Reset / Upload New File"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**About**")
    st.caption(
        "MSc IT Project — Predictive Analytics for Customer Churn\n\n"
        "Student: Prudence Jepchumba Kirui\n"
        "Supervisor: Usman Sattar"
    )


# ══════════════════════════════════════════════════════════════════════════════
# LANDING STATE — No file uploaded yet
# ══════════════════════════════════════════════════════════════════════════════

if uploaded_file is None:
    st.markdown('<p class="main-header">Telco Customer Churn Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload your dataset in the sidebar to begin the analysis.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><b>Models</b><br>5 ML classifiers compared</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><b>XAI</b><br>SHAP and LIME explanations</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><b>Segmentation</b><br>High / Medium / Low risk</div>', unsafe_allow_html=True)

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP — File has been uploaded
# ══════════════════════════════════════════════════════════════════════════════

file_bytes = uploaded_file.read()

with st.spinner("Loading and preparing data..."):
    df_raw  = load_and_clean(file_bytes)
    df_orig = pd.read_csv(io.BytesIO(file_bytes))  # Keep original for EDA display

with st.spinner("Training models — this may take a moment..."):
    X_train_scaled, y_train_res, X_test_scaled, y_test, feature_names, X_test = preprocess(df_raw)
    results, trained_models = train_all_models(X_train_scaled, y_train_res, X_test_scaled, y_test)

# Identify best model by AUC-ROC
best_model_name = max(results, key=lambda k: results[k]["AUC-ROC"])
best_model       = trained_models[best_model_name]

# Build test DataFrame for SHAP/LIME
X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)

st.markdown('<p class="main-header">Telco Customer Churn Analysis</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">Dataset: {df_raw.shape[0]:,} records | {df_raw.shape[1]} features | Best model: <span class="best-model-badge">{best_model_name}</span></p>', unsafe_allow_html=True)

# ── Top-level KPI strip ───────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
best = results[best_model_name]
k1.metric("Best Model", best_model_name)
k2.metric("AUC-ROC",    f"{best['AUC-ROC']:.4f}")
k3.metric("Recall",     f"{best['Recall']:.4f}")
k4.metric("F1-Score",   f"{best['F1-Score']:.4f}")
k5.metric("MCC",        f"{best['MCC']:.4f}")

st.markdown("---")

# ── Tab navigation ────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Data Overview",
    "Exploratory Analysis",
    "Machine Learning Models",
    "Model Explanations",
    "Churn Risk Segmentation",
    "Export",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.markdown('<p class="section-title">Dataset Overview</p>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",    f"{df_orig.shape[0]:,}")
    c2.metric("Total Columns",    df_orig.shape[1])
    c3.metric("Missing Values",   df_orig.isnull().sum().sum())
    c4.metric("Duplicate Rows",   df_orig.duplicated().sum())

    st.markdown("---")
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<p class="section-title">Data Types & Missing Values</p>', unsafe_allow_html=True)
        dtype_df = pd.DataFrame({
            "Column":        df_orig.columns,
            "Data Type":     df_orig.dtypes.astype(str).values,
            "Missing Count": df_orig.isnull().sum().values,
            "Missing %":     (df_orig.isnull().mean() * 100).round(2).astype(str).values + "%",
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    with col_right:
        st.markdown('<p class="section-title">Summary Statistics</p>', unsafe_allow_html=True)
        num_df = df_orig.select_dtypes(include="number")
        st.dataframe(num_df.describe().round(2), use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-title">Churn Distribution</p>', unsafe_allow_html=True)

    churn_counts = df_orig["Churn"].value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor("white")

    # Pie chart
    colors = ["#1E2761", "#C0392B"]
    ax1.pie(churn_counts.values, labels=churn_counts.index, autopct="%1.1f%%",
            colors=colors, startangle=90, textprops={"fontsize": 12})
    ax1.set_title("Churn Proportion", fontsize=13, fontweight="bold")

    # Bar chart
    bars = ax2.bar(churn_counts.index, churn_counts.values, color=colors, edgecolor="white", width=0.5)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 30,
                 f"{int(bar.get_height()):,}", ha="center", va="bottom", fontsize=11)
    ax2.set_title("Churn Count", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Churn")
    ax2.set_ylabel("Count")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown('<p class="section-title">Raw Data Preview</p>', unsafe_allow_html=True)
    st.dataframe(df_orig.head(50), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLORATORY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.markdown('<p class="section-title">Exploratory Data Analysis</p>', unsafe_allow_html=True)

    # Sidebar filters for this tab
    with st.sidebar:
        st.markdown("### EDA Filters")
        cat_options = [c for c in df_orig.select_dtypes("object").columns
                       if c not in ["customerID", "Churn"]]
        selected_cat = st.selectbox("Categorical feature to explore", cat_options)
        num_options  = df_orig.select_dtypes("number").columns.tolist()
        selected_num = st.selectbox("Numerical feature to explore", num_options)

    # Row 1 — Churn by contract and churn by selected categorical
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df_orig, x="Contract", hue="Churn",
                      palette={"No": "#1E2761", "Yes": "#C0392B"}, ax=ax)
        ax.set_title("Churn vs Contract Type", fontweight="bold")
        ax.set_xlabel("Contract")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with r1c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df_orig, x=selected_cat, hue="Churn",
                      palette={"No": "#1E2761", "Yes": "#C0392B"}, ax=ax)
        ax.set_title(f"Churn vs {selected_cat}", fontweight="bold")
        ax.set_xlabel(selected_cat)
        plt.xticks(rotation=25, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Row 2 — Tenure box plot and selected numerical distribution
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df_orig, x="Churn", y="tenure",
                    palette={"No": "#1E2761", "Yes": "#C0392B"}, ax=ax)
        ax.set_title("Churn vs Tenure", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with r2c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, color in [("No", "#1E2761"), ("Yes", "#C0392B")]:
            subset = df_orig[df_orig["Churn"] == label][selected_num].dropna()
            ax.hist(subset, bins=30, alpha=0.6, color=color, label=label, edgecolor="white")
        ax.set_title(f"Distribution of {selected_num} by Churn", fontweight="bold")
        ax.set_xlabel(selected_num)
        ax.legend(title="Churn")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Row 3 — Internet service and payment method
    r3c1, r3c2 = st.columns(2)

    with r3c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df_orig, x="InternetService", hue="Churn",
                      palette={"No": "#1E2761", "Yes": "#C0392B"}, ax=ax)
        ax.set_title("Churn vs Internet Service", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with r3c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df_orig, x="PaymentMethod", hue="Churn",
                      palette={"No": "#1E2761", "Yes": "#C0392B"}, ax=ax)
        ax.set_title("Churn vs Payment Method", fontweight="bold")
        ax.set_xlabel("Payment Method")
        plt.xticks(rotation=20, ha="right")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Row 4 — Monthly charges violin and correlation matrix
    r4c1, r4c2 = st.columns(2)

    with r4c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.violinplot(data=df_orig, x="Churn", y="MonthlyCharges",
                       palette={"No": "#1E2761", "Yes": "#C0392B"}, ax=ax)
        ax.set_title("Monthly Charges Distribution by Churn", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with r4c2:
        num_df = df_orig.select_dtypes(include="number")
        corr   = num_df.corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues",
                    linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
        ax.set_title("Correlation Matrix (Numerical Features)", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MACHINE LEARNING MODELS
# ══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.markdown('<p class="section-title">Model Performance Comparison</p>', unsafe_allow_html=True)

    # Build summary table
    summary_rows = []
    for name, r in results.items():
        row = {
            "Model":     name,
            "Accuracy":  r["Accuracy"],
            "Precision": r["Precision"],
            "Recall":    r["Recall"],
            "F1-Score":  r["F1-Score"],
            "AUC-ROC":   r["AUC-ROC"],
            "MCC":       r["MCC"],
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values("AUC-ROC", ascending=False)
    summary_df["Best"] = summary_df["Model"].apply(lambda m: "Best" if m == best_model_name else "")

    def highlight_best(row):
        if row["Model"] == best_model_name:
            return ["background-color: #e8f0fe; font-weight: bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        summary_df.style.apply(highlight_best, axis=1).format({
            "Accuracy": "{:.4f}", "Precision": "{:.4f}",
            "Recall":   "{:.4f}", "F1-Score":  "{:.4f}",
            "AUC-ROC":  "{:.4f}", "MCC":       "{:.4f}",
        }),
        use_container_width=True, hide_index=True
    )

    st.success(f"Best Model: **{best_model_name}** with AUC-ROC = {results[best_model_name]['AUC-ROC']:.4f}")

    st.markdown("---")
    st.markdown('<p class="section-title">Confusion Matrices</p>', unsafe_allow_html=True)

    model_names = list(results.keys())
    colors_cm   = ["Blues", "Greens", "Oranges", "Purples", "Reds"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.patch.set_facecolor("white")

    for ax, name, cmap in zip(axes, model_names, colors_cm):
        cm_val = confusion_matrix(y_test, results[name]["preds"])
        sns.heatmap(cm_val, annot=True, fmt="d", cmap=cmap, ax=ax,
                    xticklabels=["No Churn", "Churn"],
                    yticklabels=["No Churn", "Churn"],
                    linewidths=0.5)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices — All Five Models", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    cm_bytes = fig_to_bytes(fig)
    plt.close()

    st.markdown("---")
    st.markdown('<p class="section-title">ROC Curves</p>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    roc_colors = ["#1E2761", "#27AE60", "#E67E22", "#8E44AD", "#C0392B"]

    for name, color in zip(model_names, roc_colors):
        fpr, tpr, _ = roc_curve(y_test, results[name]["probs"])
        lw = 2.5 if name == best_model_name else 1.5
        ax.plot(fpr, tpr, color=color, lw=lw,
                label=f"{name} (AUC = {results[name]['AUC-ROC']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve Comparison — All Five Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    roc_bytes = fig_to_bytes(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL EXPLANATIONS
# ══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.markdown(f'<p class="section-title">SHAP Explanations — {best_model_name}</p>', unsafe_allow_html=True)

    with st.spinner("Computing SHAP values..."):
        explainer   = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test_df)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    col_bar, col_bee = st.columns(2)

    with col_bar:
        st.markdown("**Global Feature Importance (SHAP Bar)**")
        fig, ax = plt.subplots(figsize=(7, 6))
        shap.summary_plot(sv, X_test_df, plot_type="bar",
                          show=False, max_display=15, color="#1E2761")
        ax.set_title(f"SHAP Global Importance — {best_model_name}", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        shap_bar_bytes = fig_to_bytes(fig)
        plt.close()

    with col_bee:
        st.markdown("**Feature Impact Direction (SHAP Beeswarm)**")
        fig, ax = plt.subplots(figsize=(7, 6))
        shap.summary_plot(sv, X_test_df, show=False, max_display=15)
        ax.set_title(f"SHAP Beeswarm — {best_model_name}", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        shap_bee_bytes = fig_to_bytes(fig)
        plt.close()

    st.markdown("---")
    st.markdown("**SHAP Waterfall — Individual Customer**")

    max_row    = len(X_test_df) - 1
    sample_idx = st.slider("Select customer row (test set index)", 0, max_row, 0)

    base_val = (explainer.expected_value[1]
                if isinstance(explainer.expected_value, list)
                else explainer.expected_value)

    shap_exp = shap.Explanation(
        values        = sv[sample_idx],
        base_values   = base_val,
        data          = X_test_df.iloc[sample_idx].values,
        feature_names = feature_names
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.waterfall_plot(shap_exp, show=False, max_display=15)
    plt.title(f"SHAP Waterfall — Customer #{sample_idx} | Actual Churn: {int(y_test.iloc[sample_idx])}",
              fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    shap_wf_bytes = fig_to_bytes(fig)
    plt.close()

    st.markdown("---")
    st.markdown("**LIME Explanation — Individual Customer**")

    lime_row = st.slider("Select customer row for LIME", 0, max_row, 0, key="lime_slider")

    with st.spinner("Generating LIME explanation..."):
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data = X_train_scaled,
            feature_names = feature_names,
            class_names   = ["No Churn", "Churn"],
            mode          = "classification",
            random_state  = 42
        )
        lime_exp = lime_explainer.explain_instance(
            data_row  = X_test_df.iloc[lime_row].values,
            predict_fn = best_model.predict_proba,
            num_features = 10
        )

    fig = lime_exp.as_pyplot_figure()
    plt.title(f"LIME Explanation — Customer #{lime_row} | Actual Churn: {int(y_test.iloc[lime_row])}",
              fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    lime_bytes = fig_to_bytes(fig)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CHURN RISK SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.markdown('<p class="section-title">Customer Churn Risk Segmentation</p>', unsafe_allow_html=True)

    # Apply probability thresholds
    best_probs = results[best_model_name]["probs"]

    def risk_label(p):
        if p >= 0.70:
            return "High Risk"
        elif p >= 0.40:
            return "Medium Risk"
        return "Low Risk"

    risk_df = pd.DataFrame({
        "Customer Index":    y_test.index,
        "Churn Probability": best_probs.round(4),
        "Actual Churn":      y_test.values,
        "Risk Level":        [risk_label(p) for p in best_probs],
    })

    # Summary strip
    risk_counts = risk_df["Risk Level"].value_counts()
    s1, s2, s3 = st.columns(3)
    s1.metric("High Risk",   risk_counts.get("High Risk",   0), help="Probability >= 70%")
    s2.metric("Medium Risk", risk_counts.get("Medium Risk", 0), help="Probability 40–70%")
    s3.metric("Low Risk",    risk_counts.get("Low Risk",    0), help="Probability < 40%")

    st.markdown("---")
    col_chart, col_filter = st.columns([2, 1])

    with col_chart:
        fig, ax = plt.subplots(figsize=(7, 4))
        bar_colors = {"High Risk": "#C0392B", "Medium Risk": "#E67E22", "Low Risk": "#1E2761"}
        ordered    = ["High Risk", "Medium Risk", "Low Risk"]
        counts     = [risk_counts.get(r, 0) for r in ordered]
        bars       = ax.bar(ordered, counts,
                            color=[bar_colors[r] for r in ordered],
                            edgecolor="white", width=0.5)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 4,
                    str(int(bar.get_height())),
                    ha="center", va="bottom", fontsize=12, fontweight="bold")
        ax.set_title("Customer Churn Risk Distribution", fontsize=13, fontweight="bold")
        ax.set_xlabel("Risk Level")
        ax.set_ylabel("Number of Customers")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        seg_chart_bytes = fig_to_bytes(fig)
        plt.close()

    with col_filter:
        st.markdown("**Filter by Risk Level**")
        filter_choice = st.multiselect(
            "Select risk level(s)",
            options=["High Risk", "Medium Risk", "Low Risk"],
            default=["High Risk", "Medium Risk", "Low Risk"]
        )
        prob_range = st.slider(
            "Churn probability range",
            min_value=0.0, max_value=1.0,
            value=(0.0, 1.0), step=0.01
        )

    # Apply filters
    filtered_df = risk_df[
        risk_df["Risk Level"].isin(filter_choice) &
        risk_df["Churn Probability"].between(*prob_range)
    ]

    st.markdown(f"**Showing {len(filtered_df):,} customers**")
    st.dataframe(
        filtered_df.style.background_gradient(subset=["Churn Probability"], cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True
    )

    st.download_button(
        label     = "Download Segmented Data (CSV)",
        data      = df_to_csv(filtered_df),
        file_name = "churn_risk_segmentation.csv",
        mime      = "text/csv"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.markdown('<p class="section-title">Export Results</p>', unsafe_allow_html=True)

    # Build full predictions dataframe
    preds_df = pd.DataFrame({
        "Customer Index":        y_test.index,
        "Actual Churn":          y_test.values,
        "Predicted Churn":       results[best_model_name]["preds"],
        "Churn Probability":     results[best_model_name]["probs"].round(4),
        "Risk Level":            [risk_label(p) for p in results[best_model_name]["probs"]],
    })

    st.markdown("### Data Exports")
    e1, e2, e3 = st.columns(3)

    with e1:
        st.markdown("**Cleaned Dataset**")
        st.caption("Preprocessed CSV with all transformations applied.")
        st.download_button(
            "Download Cleaned Data",
            data      = df_to_csv(df_raw),
            file_name = "telco_churn_cleaned.csv",
            mime      = "text/csv",
            use_container_width=True
        )

    with e2:
        st.markdown("**Model Predictions**")
        st.caption(f"Predictions from best model: {best_model_name}")
        st.download_button(
            "Download Predictions",
            data      = df_to_csv(preds_df),
            file_name = "churn_predictions.csv",
            mime      = "text/csv",
            use_container_width=True
        )

    with e3:
        st.markdown("**Performance Summary**")
        st.caption("All model metrics in one CSV.")
        st.download_button(
            "Download Model Metrics",
            data      = df_to_csv(summary_df.drop(columns=["Best"])),
            file_name = "model_performance_summary.csv",
            mime      = "text/csv",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### Chart Exports")

    img1, img2, img3, img4, img5 = st.columns(5)

    with img1:
        st.download_button("Confusion Matrices", cm_bytes,
                           "confusion_matrices.png", "image/png", use_container_width=True)
    with img2:
        st.download_button("ROC Curves", roc_bytes,
                           "roc_curves.png", "image/png", use_container_width=True)
    with img3:
        st.download_button("SHAP Bar Plot", shap_bar_bytes,
                           "shap_bar.png", "image/png", use_container_width=True)
    with img4:
        st.download_button("SHAP Beeswarm", shap_bee_bytes,
                           "shap_beeswarm.png", "image/png", use_container_width=True)
    with img5:
        st.download_button("Risk Segmentation", seg_chart_bytes,
                           "risk_segmentation.png", "image/png", use_container_width=True)

    st.markdown("---")
    st.markdown("### Predictions Preview")
    st.dataframe(preds_df.head(50), use_container_width=True, hide_index=True)
