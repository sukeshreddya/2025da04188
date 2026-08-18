import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

st.set_page_config(layout="wide")

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

st.title("ML Models Comparison — Assignment 2 Starter")

st.markdown(
    """
    <style>
    .section-card {
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 12px;
        padding: 1rem 1.15rem 1rem 1.15rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
        margin-bottom: 1.1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    }
    .section-card h3, .section-card h4 {
        margin-top: 0;
        margin-bottom: 0.75rem;
        font-weight: 650;
        letter-spacing: -0.02em;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.55rem;
        margin-bottom: 0.7rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 0.45rem 0.9rem;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .stButton > button {
        border-radius: 9px;
        padding: 0.55rem 1rem;
        font-weight: 600;
    }
    .stDataFrame {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

df = None
right_col, left_col = st.columns([0.9, 1.3])

with right_col:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Training Configuration")
        target = st.selectbox("Select target column", options=df.columns)
        test_size = st.slider("Test size (fraction)", 0.1, 0.5, 0.35)
        random_state = st.number_input("Random state", value=42)

        if st.button("Train & Evaluate", use_container_width=True):
            try:
                X = df.drop(columns=[target])
                y = df[target]

                X = X.fillna(X.mean(numeric_only=True))
                y = y.dropna()
                X = X.loc[y.index]

                X = pd.get_dummies(X, drop_first=True)
                if X.empty or len(X) == 0:
                    st.error("Error: No valid features after preprocessing.")
                    st.stop()

                if y.dtype == object or y.dtype.name == "category":
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                else:
                    y = y.astype(int)

                if len(np.unique(y)) < 2:
                    st.error("Error: Target variable must have at least 2 classes.")
                    st.stop()

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y
                )

                n_neighbors = min(5, max(1, len(X_train) - 1))
                models = {
                    "LogisticRegression": LogisticRegression(max_iter=1000),
                    "DecisionTree": DecisionTreeClassifier(),
                    "KNN": KNeighborsClassifier(n_neighbors=n_neighbors),
                    "GaussianNB": GaussianNB(),
                    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
                }

                results = []
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    try:
                        if len(np.unique(y)) == 2:
                            y_proba = model.predict_proba(X_test)[:, 1]
                            auc = roc_auc_score(y_test, y_proba)
                        else:
                            auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr", average="macro")
                    except Exception:
                        auc = np.nan

                    res = {
                        "model": name,
                        "accuracy": accuracy_score(y_test, y_pred),
                        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                        "auc": auc,
                        "mcc": matthews_corrcoef(y_test, y_pred),
                    }
                    results.append(res)
                    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))

                results_df = pd.DataFrame(results).set_index("model")
                st.session_state.results_df = results_df
                st.session_state.models = models
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.selected_model = list(models.keys())[0]
                st.success("Training and evaluation complete — models saved in model/ directory.")
            except Exception as e:
                st.error(f"Error during training: {str(e)}")
                import traceback
                st.write(traceback.format_exc())
        st.markdown("</div>", unsafe_allow_html=True)

with left_col:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Dataset Overview")
    if df is None:
        st.info(
            """
            **Instructions:**
            1. Upload a CSV file with at least 20 rows for meaningful results
            2. Select the target column
            3. Adjust test size and random state
            4. Click "Train & Evaluate" to run all 5 models

            **Recommended:** Use the preprocessed credit card dataset for best results.
            """
        )
    else:
        info_col1, info_col2 = st.columns(2)
        info_col1.metric("Rows", f"{len(df):,}")
        info_col2.metric("Columns", f"{len(df.columns):,}")
        st.write("Target candidates:", ", ".join(df.columns.tolist()))
        if len(df) < 20:
            st.warning(f"⚠️ Dataset has only {len(df)} rows. Recommend at least 20 rows for meaningful results.")

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

if "results_df" in st.session_state and st.session_state.results_df is not None:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Results")
    result_left, result_right = st.columns([1.2, 1])
    with result_left:
        st.dataframe(st.session_state.results_df, use_container_width=True)
    with result_right:
        model_names = list(st.session_state.models.keys())
        tabs = st.tabs(model_names)
        for tab, model_name in zip(tabs, model_names):
            with tab:
                cm_model = st.session_state.models[model_name]
                y_pred_sel = cm_model.predict(st.session_state.X_test)
                cm = confusion_matrix(st.session_state.y_test, y_pred_sel)
                fig, ax = plt.subplots(figsize=(4.5, 3.4))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                fig.tight_layout()
                st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

METRICS_PATH = os.path.join("results", "model_metrics.csv")
if os.path.exists(METRICS_PATH):
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.subheader("Saved Model Results")
    try:
        metrics_df = pd.read_csv(METRICS_PATH, index_col=0)
        st.write("Overall metrics (loaded from results/model_metrics.csv)")
        st.dataframe(metrics_df, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load saved metrics: {e}")
    st.markdown("</div>", unsafe_allow_html=True)
