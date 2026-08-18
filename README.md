Machine Learning Assignment 2

## Problem statement
(Add a short description of the classification problem you choose.)

## Dataset
- Chosen dataset: Credit Card Fraud Detection (Kaggle)
- Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- Description: Binary classification (fraud / non-fraud), 30 numeric features (PCA transformed), 284,807 instances, highly imbalanced. Good for practicing imbalance handling and evaluation metrics.

## Models implemented
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbor
- Naive Bayes (Gaussian/Multinomial)
- Random Forest (Ensemble)

## How to run
1. Create a virtual environment and install requirements:

```bash
pip install -r requirements.txt
```

2. Run Streamlit app:

```bash
streamlit run app.py
```

## Repository structure
project-folder/
- app.py
- requirements.txt
- README.md
- test_data.csv
- model/ (saved models)

## Notes
- Add dataset description, model comparison table, observations and links before submission.
