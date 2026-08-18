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

uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"]) 
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data preview")
    st.dataframe(df.head())

    target = st.selectbox("Select target column", options=df.columns)
    test_size = st.slider("Test size (fraction)", 0.1, 0.5, 0.2)
    random_state = st.number_input("Random state", value=42)

    if st.button("Train & Evaluate"):
        try:
            X = df.drop(columns=[target])
            y = df[target]

            # Handle missing values
            X = X.fillna(X.mean(numeric_only=True))
            y = y.dropna()
            X = X.loc[y.index]

            # Basic preprocessing
            X = pd.get_dummies(X, drop_first=True)
            if X.empty or len(X) == 0:
                st.error("Error: No valid features after preprocessing.")
                st.stop()

            # Encode target if needed
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

            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "DecisionTree": DecisionTreeClassifier(),
                "KNN": KNeighborsClassifier(),
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

                # save model with sanitized name
                joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))

            results_df = pd.DataFrame(results).set_index("model")
            st.write("Evaluation metrics")
            st.dataframe(results_df)

            # show confusion matrix for selected model
            sel = st.selectbox("Show confusion matrix for", options=list(models.keys()))
            cm_model = joblib.load(os.path.join(MODEL_DIR, f"{sel}.pkl"))
            y_pred_sel = cm_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred_sel)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

            st.success("Training and evaluation complete — models saved in model/ directory.")
        except Exception as e:
            st.error(f"Error during training: {str(e)}")
            import traceback
            st.write(traceback.format_exc())

        st.success("Training and evaluation complete — models saved in model/ directory.")
else:
    st.info("Upload a CSV to get started. A sample `test_data.csv` placeholder is included in the repo.")

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
