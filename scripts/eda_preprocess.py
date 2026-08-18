"""EDA and preprocessing helper.

Usage:
  python scripts/eda_preprocess.py --input test_data.csv --output_dir data/

The script computes basic statistics, class balance, missing values,
saves a correlation heatmap, and writes a preprocessed CSV.
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def eda_and_preprocess(input_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_path)

    report_lines = []
    report_lines.append(f"Input file: {input_path}")
    report_lines.append(f"Rows, columns: {df.shape}")
    report_lines.append("\nColumn types:")
    report_lines.append(str(df.dtypes))
    report_lines.append("\nMissing values per column:")
    report_lines.append(str(df.isnull().sum()))
    report_lines.append("\nBasic statistics:")
    report_lines.append(str(df.describe()))

    # Identify target if named 'Class' or 'target' else prompt
    target_col = None
    for candidate in ["Class", "class", "target", "fraud"]:
        if candidate in df.columns:
            target_col = candidate
            break

    if target_col is None:
        # fallback: last column
        target_col = df.columns[-1]

    report_lines.append(f"\nAssumed target column: {target_col}")
    report_lines.append("\nClass distribution:")
    report_lines.append(str(df[target_col].value_counts()))

    # Correlation heatmap (for numeric columns only)
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()
    report_lines.append(f"\nCorrelation heatmap saved to: {heatmap_path}")

    # Simple preprocessing: scale numeric features except target
    feat_cols = [c for c in numeric.columns if c != target_col]
    if len(feat_cols) > 0:
        scaler = StandardScaler()
        numeric_scaled = numeric.copy()
        numeric_scaled[feat_cols] = scaler.fit_transform(numeric[feat_cols])
        preprocessed = df.copy()
        for c in feat_cols:
            preprocessed[c] = numeric_scaled[c]
        out_path = os.path.join(output_dir, "preprocessed.csv")
        preprocessed.to_csv(out_path, index=False)
        report_lines.append(f"\nPreprocessed CSV saved to: {out_path}")
    else:
        report_lines.append("\nNo numeric features to scale; skipping preprocessing save.")

    report_path = os.path.join(output_dir, "eda_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(report_lines))

    print("EDA and preprocessing complete.")
    print(f"Report: {report_path}")
    print(f"Heatmap: {heatmap_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV file")
    parser.add_argument("--output_dir", default="data/", help="Output folder")
    args = parser.parse_args()
    eda_and_preprocess(args.input, args.output_dir)


if __name__ == "__main__":
    main()
