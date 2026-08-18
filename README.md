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

## Models Implemented
1. **Logistic Regression** — Linear baseline model
2. **Decision Tree Classifier** — Tree-based single model
3. **K-Nearest Neighbor** — Instance-based learning
4. **Gaussian Naive Bayes** — Probabilistic baseline
5. **Random Forest** — Ensemble method (100 trees)

## Model Comparison & Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | AUC Score | MCC |
|-------|----------|-----------|--------|----------|-----------|-----|
| Logistic Regression | 0.9746 | 0.0608 | 0.9184 | 0.1141 | 0.9721 | 0.2329 |
| Decision Tree | 0.9989 | 0.6762 | 0.7245 | 0.6995 | 0.8619 | 0.6994 |
| **K-Nearest Neighbor** | **0.9995** | **0.9186** | **0.8061** | **0.8587** | **0.9437** | **0.8603** |
| Gaussian Naive Bayes | 0.9764 | 0.0588 | 0.8469 | 0.1099 | 0.9632 | 0.2195 |
| Random Forest | 0.9995 | 0.9605 | 0.7449 | 0.8391 | 0.9529 | 0.8456 |

**Test Set Size:** 20% of total data (56,961 samples)  
**Random State:** 42 (for reproducibility)

### Model Performance Observations

#### 1. **Logistic Regression**
- **Strength:** High AUC (0.9721) and very high recall (0.9184), indicating excellent detection of fraud cases
- **Weakness:** Extremely low precision (0.0608) leading to many false positives; high false alarm rate makes this model impractical for deployment
- **Use Case:** Could be used as a first-pass filter, but not suitable as primary model due to false positive burden

#### 2. **Decision Tree**
- **Strength:** Excellent accuracy (0.9989) and balanced recall (0.7245) with F1-score of 0.6995
- **Weakness:** Prone to overfitting on this dataset; lower precision (0.6762) indicates room for improvement
- **Use Case:** Good interpretability and reasonable performance, but not optimal for production use

#### 3. **K-Nearest Neighbor** ⭐ **WINNER**
- **Strength:** Best overall performance with highest F1-score (0.8587), highest precision (0.9186), and highest MCC (0.8603)
- **Characteristics:** Excellent balance between false positives and false negatives; reliable predictions across all metrics
- **Weakness:** Slightly lower recall (0.8061) means ~19.4% of fraud cases may be missed, but compensated by very high precision
- **Use Case:** Optimal for fraud detection where false positives are costly; recommended for production deployment
- **Advantage:** Simple, interpretable, and highly effective on this imbalanced dataset

#### 4. **Gaussian Naive Bayes**
- **Strength:** High AUC (0.9632) indicating good ranking of probabilities; high recall (0.8469)
- **Weakness:** Extremely low precision (0.0588), similar to Logistic Regression; not practical due to overwhelming false positives
- **Use Case:** Suitable only for early-stage detection with downstream verification

#### 5. **Random Forest**
- **Strength:** Highest precision (0.9605) and excellent accuracy (0.9995); second-best F1-score (0.8391)
- **Weakness:** Slightly lower recall (0.7449) compared to KNN; lower MCC (0.8456) vs KNN
- **Use Case:** Strong choice for production when false positives must be minimized; trades off recall for extremely high precision
- **Ensemble Benefit:** Reduces overfitting risk compared to Decision Tree

### Overall Winner & Recommendation

**🏆 K-Nearest Neighbor (KNN)** is the best-performing model for this fraud detection task.

**Rationale:**
- **Balanced Performance:** KNN achieves the best F1-score (0.8587), which balances precision and recall optimally
- **Highest MCC:** MCC of 0.8603 is superior to all competitors, indicating strongest overall predictive power
- **Reliable Precision:** With 91.86% precision, KNN minimizes false positives while maintaining reasonable recall
- **Practical Deployment:** False positives are costly in banking; KNN's high precision makes it most suitable for real-world use
- **Interpretability:** Instance-based approach is easy to understand and debug

**Secondary Choice:** Random Forest can be used if minimizing false positives is prioritized above all else (highest precision at 96.05%).

## How to Run

1. Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit application:

```bash
streamlit run app.py
```

3. Upload your CSV dataset in the web interface and select target column to train and evaluate all models.

4. To retrain models on preprocessed credit card data:

```bash
python scripts/train_models.py --input data/eda_creditcard/preprocessed.csv
```

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
├── data/
│   ├── creditcard.csv             # Original dataset
│   ├── preprocessed.csv           # Preprocessed dataset
│   └── eda_creditcard/
│       ├── preprocessed.csv       # Final preprocessed data
│       └── eda_report.txt         # EDA insights
├── model/
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

✅ **Dataset Upload** — Upload CSV file for training
✅ **Model Selection** — Choose from 5 trained models
✅ **Evaluation Metrics** — View accuracy, precision, recall, F1, AUC, MCC
✅ **Confusion Matrix** — Visualize model predictions

## Notes

- Dataset is highly imbalanced; evaluation focuses on metrics beyond accuracy
- PCA-transformed features protect original sensitive information
- Stratified train-test split ensures representative class distribution
- All models trained with random_state=42 for reproducibility
