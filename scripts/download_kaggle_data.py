"""
Download Credit Card Fraud dataset from Kaggle using the Kaggle API.

Usage:
1. Install dependencies: `pip install -r requirements.txt`
2. Configure Kaggle credentials: place `kaggle.json` in ~/.kaggle/ or set env vars `KAGGLE_USERNAME` and `KAGGLE_KEY`.
3. Run: `python scripts/download_kaggle_data.py --output data/creditcard.csv`

Note: On BITS Virtual Lab, ensure Kaggle API access is allowed and credentials are set.
"""
import argparse
import os
import subprocess


def download(output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Uses kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p <dir>
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "mlg-ulb/creditcardfraud",
        "--unzip",
        "-p",
        os.path.dirname(output_path),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/creditcard.csv", help="Output CSV path")
    args = parser.parse_args()
    download(args.output)


if __name__ == "__main__":
    main()
