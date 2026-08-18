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

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

st.title("ML Models Comparison — Assignment 2 Starter")

st.markdown("""
**Instructions:**
1. Upload a CSV file with at least 20 rows for meaningful results
2. Select the target column
3. Adjust test size and random state
4. Click "Train & Evaluate" to run all 5 models

**Recommended:** Use the preprocessed credit card dataset for best results.
""")

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"]) 
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Warning for small datasets
    if len(df) < 20:
        st.warning(f"⚠️ Dataset has only {len(df)} rows. Recommend at least 20 rows for meaningful results.")
    
    st.write("Data preview")
    st.dataframe(df.head())

    target = st.selectbox("Select target column", options=df.columns)
    test_size = st.slider("Test size (fraction)", 0.1, 0.5, 0.2)
    random_state = st.number_input("Random state", value=42)

    if st.button("Train & Evaluate"):
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

    if "results_df" in st.session_state and st.session_state.results_df is not None:
        st.write("Evaluation metrics")
        st.dataframe(st.session_state.results_df)

        model_names = list(st.session_state.models.keys())
        selected_model = st.selectbox(
            "Show confusion matrix for",
            options=model_names,
            index=model_names.index(st.session_state.get("selected_model", model_names[0])),
        )
        st.session_state.selected_model = selected_model

        cm_model = st.session_state.models[selected_model]
        y_pred_sel = cm_model.predict(st.session_state.X_test)
        cm = confusion_matrix(st.session_state.y_test, y_pred_sel)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
else:
    st.info("""
    **How to get started:**
    
    Option 1: Use the test_data.csv from the repo (simple example)
    Option 2: For real results, download the Credit Card Fraud Detection dataset:
    - Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    - Download creditcard.csv
    - Upload it here
    - Select 'Class' as target column
    - Run Train & Evaluate
    """)

# Section: show saved results if available
METRICS_PATH = os.path.join("results", "model_metrics.csv")
if os.path.exists(METRICS_PATH):
    st.header("Saved model results")
    try:
        metrics_df = pd.read_csv(METRICS_PATH, index_col=0)
        st.write("Overall metrics (loaded from results/model_metrics.csv)")
        st.dataframe(metrics_df)

        sel_model = st.selectbox("Select saved model to view confusion matrix", options=list(metrics_df.index))
        cm_path = os.path.join("results", f"cm_{sel_model}.csv")
        if os.path.exists(cm_path):
            cm = pd.read_csv(cm_path, header=None).values
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
        else:
            st.info(f"Confusion matrix file not found for {sel_model}: {cm_path}")
    except Exception as e:
        st.error(f"Failed to load saved metrics: {e}")
