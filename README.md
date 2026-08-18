Machine Learning Assignment 2 — Credit Card Fraud Detection

## Problem Statement

Credit card fraud is a critical problem in the financial industry, causing billions in losses annually. The objective of this assignment is to build and compare multiple classification models to accurately identify fraudulent transactions from a highly imbalanced dataset. Early detection of fraudulent activities can prevent financial losses, protect customers, and maintain the integrity of the financial system.

**Classification Problem:** Binary classification to predict whether a credit card transaction is fraudulent (Class=1) or legitimate (Class=0).

## Dataset Description

**Dataset:** Credit Card Fraud Detection (from Kaggle)
- **Source:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Size:** 284,807 transactions with 30 features
- **Features:** 28 PCA-transformed numerical features (V1-V28), Amount, and Time
- **Target Variable:** Class (0=Legitimate, 1=Fraud)
- **Class Distribution:** Highly imbalanced (~0.17% fraud, ~99.83% legitimate)
- **Preprocessing:** Dataset was already preprocessed with PCA transformations for privacy. Minor EDA and standardization applied before model training.

The imbalanced nature of this dataset makes it ideal for practicing evaluation metrics beyond simple accuracy, such as AUC, Precision, Recall, F1-Score, and Matthews Correlation Coefficient (MCC).

## GitHub Repository Link
**Repository:** https://github.com/sukeshreddya/2025da04188

## Live Streamlit App Link
**App URL:** https://2025da04188-ml-assignment-2.streamlit.app/

## Models Implemented
1. **Logistic Regression** — Linear baseline model
2. **Decision Tree Classifier** — Tree-based single model
3. **K-Nearest Neighbor** — Instance-based learning
4. **Gaussian Naive Bayes** — Probabilistic baseline
5. **Random Forest** — Ensemble method (100 trees)

## Model Comparison & Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | AUC Score | MCC |
|-------|----------|-----------|--------|----------|-----------|-----|
| Logistic Regression | 0.9987 | 0.8766 | 0.7947 | 0.8306 | 0.9610 | 0.6662 |
| Decision Tree | 0.9993 | 0.9274 | 0.9165 | 0.9219 | 0.9165 | 0.8439 |
| K-Nearest Neighbor | 0.9977 | 0.4989 | 0.5000 | 0.4994 | 0.6203 | -0.0003 |
| Gaussian Naive Bayes | 0.9854 | 0.5603 | 0.9351 | 0.6026 | 0.9793 | 0.3239 |
| **Random Forest** | **0.9996** | **0.9661** | **0.9422** | **0.9538** | **0.9794** | **0.9080** |

**Test Set Size:** ~35% of total data  
**Random State:** 42 (for reproducibility)

### Model Performance Observations

#### 1. **Logistic Regression**
- **Strength:** High AUC (0.9610) and solid F1-score (0.8306) with balanced precision (0.8766) and recall (0.7947)
- **Weakness:** Slightly lower recall means it misses more fraud cases compared to ensemble methods
- **Use Case:** Good baseline model; provides stable performance across metrics; suitable when interpretability is important

#### 2. **Decision Tree**
- **Strength:** Excellent balance with precision (0.9274), recall (0.9165), and F1-score (0.9219); easy to interpret
- **Weakness:** Prone to overfitting; lower AUC (0.9165) compared to Random Forest; MCC of 0.8439 indicates room for improvement
- **Use Case:** Interpretable model useful for understanding decision boundaries; good for feature importance analysis

#### 3. **K-Nearest Neighbor**
- **Strength:** Very high accuracy (0.9977)
- **Weakness:** **Severe performance degradation** - precision drops to 49.89%, recall is only 50%, and MCC near 0 (-0.0003) indicating random guessing
- **Critical Issue:** Model essentially defaults to predicting majority class and fails to distinguish fraud patterns
- **Recommendation:** **Not suitable for production** - model performs no better than random classification on this dataset

#### 4. **Gaussian Naive Bayes**
- **Strength:** Highest recall (0.9351) among single models; very high AUC (0.9793) indicating excellent probability ranking
- **Weakness:** Low precision (0.5603) results in excessive false positives; moderate F1-score (0.6026)
- **Use Case:** Could serve as a pre-screening filter where catching all fraud is prioritized, with subsequent verification required

#### 5. **Random Forest** **WINNER**
- **Strength:** Best overall performance with highest MCC (0.9080), highest precision (0.9661), highest recall (0.9422), and best F1-score (0.9538)
- **Characteristics:** Exceptional balance between false positives and false negatives; robust ensemble reduces variance
- **Advantage:** Handles imbalanced data well through ensemble voting; maintains high performance across all metrics
- **Use Case:** **Recommended for production deployment** - minimizes both types of errors effectively
- **Why Superior:** Random Forest's ensemble approach captures complex fraud patterns better than single models

### Overall Winner & Recommendation

**Random Forest** is the clear best-performing model for this fraud detection task.

**Rationale:**
- **Highest MCC (0.9080):** Superior predictive power compared to all alternatives
- **Exceptional Precision (0.9661):** Minimizes costly false positives in banking operations
- **Best Recall (0.9422):** Captures 94.22% of actual fraud cases with minimal misses
- **Best F1-Score (0.9538):** Achieves optimal balance between precision and recall
- **Robust Ensemble Method:** Reduces overfitting risk; naturally handles imbalanced data through voting mechanism
- **Production-Ready:** Most reliable model for real-world fraud detection deployment

**Model Ranking (Best to Worst):**
1. **Random Forest** — MCC: 0.9080  Production Ready
2. **Decision Tree** — MCC: 0.8439 (Good, but less robust)
3. **Logistic Regression** — MCC: 0.6662 (Acceptable baseline)
4. **Gaussian Naive Bayes** — MCC: 0.3239 (High recall, too many false alarms)
5. **K-Nearest Neighbor** — MCC: -0.0003 (Not suitable; random performance)

## How to Run

1. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:

```bash
streamlit run app.py
```

### Option A: Quick Test (test_data.csv)
- Upload `test_data.csv` from the repo
- Select `target` as target column
- **Important:** Set test_size to **0.2**
- Click "Train & Evaluate"
- **Note:** Small dataset gives quick results but less meaningful metrics

### Option B: Real Results (creditcard.csv - Recommended)
- Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Upload to app
- Select `Class` as target column
- Keep test_size at 0.35
- Click "Train & Evaluate"
- **Best results matching the metrics in this README**

## UI Flow: From Upload to Model Comparison

The application follows a simple user workflow for quick evaluation and comparison:

```text
Upload CSV
   ↓
Preview dataset
   ↓
Select target column
   ↓
Adjust test size and random state
   ↓
Click "Train & Evaluate"
   ↓
All 5 models train and evaluate
   ↓
Metrics table is displayed
   ↓
Choose model from dropdown
   ↓
Confusion matrix updates for selected model
```

This flow is implemented in the Streamlit app located in `app.py` and allows users to:
- upload a dataset,
- inspect the data preview,
- choose the target variable,
- train all machine learning models together,
- compare model performance with key metrics,
- and inspect the confusion matrix for any model.

## How the App Works After Upload

Once a CSV file is uploaded, the app reads the dataset into a Pandas DataFrame and shows a preview. The user then selects the target column, configures the train/test split ratio and random state, and clicks "Train & Evaluate". The app separates the data into features (`X`) and target (`y`), fills missing values, encodes non-numeric fields, and converts the target to numeric labels when needed. It then splits the data into training and test sets, trains all five classifiers, and evaluates each model on the same unseen test set.

The evaluation step calculates accuracy, precision, recall, F1-score, AUC, and MCC for every model. These values are displayed in a comparison table so the user can immediately compare model performance. After that, the user can choose any trained model from a dropdown and the app generates its confusion matrix, making it easy to inspect true positives, false positives, false negatives, and true negatives for the selected classifier.

This means the app is not just loading a file; it is converting the uploaded data into a machine learning workflow: dataset preparation → train/test split → model training → metrics comparison → confusion matrix analysis.

## Repository Structure

```
Mac_Learning_Assignment2/
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── assignment.txt                  # Assignment instructions
├── test_data.csv                   # Sample test data
├── scripts/
│   ├── download_kaggle_data.py    # Download credit card dataset
│   ├── eda_preprocess.py          # EDA and preprocessing
│   └── train_models.py            # Model training script
├── model/
│   ├── train_models.py
│   ├── LogisticRegression.pkl
│   ├── DecisionTree.pkl
│   ├── KNN.pkl
│   ├── GaussianNB.pkl
│   └── RandomForest.pkl
└── results/
    ├── model_metrics.csv          # Performance metrics for all models
    └── cm_*.csv                   # Confusion matrices
```

## Key Dependencies

- `streamlit` — Interactive web application framework
- `scikit-learn` — ML models and evaluation metrics
- `pandas` — Data manipulation and analysis
- `numpy` — Numerical operations
- `matplotlib` & `seaborn` — Visualization
- `joblib` — Model serialization

## Streamlit App Features

**Dataset Upload** — Upload CSV file for training
**Model Selection** — Choose from 5 trained models
**Evaluation Metrics** — View accuracy, precision, recall, F1, AUC, MCC
**Confusion Matrix** — Visualize model predictions

## Notes

- Dataset is highly imbalanced; evaluation focuses on metrics beyond accuracy
- PCA-transformed features protect original sensitive information
- Stratified train-test split ensures representative class distribution
- All models trained with random_state=42 for reproducibility
