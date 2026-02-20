
import subprocess, sys

# Install xgboost at runtime — it is NOT pre-installed in the SKLearn processing container
subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost==1.5.2", "--quiet"])

import os, json, tarfile, glob, traceback
import pandas as pd, numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score
)

def list_dir(path):
    """Print directory contents for debugging."""
    if os.path.isdir(path):
        print(f"  Contents of {path}: {os.listdir(path)}")
    else:
        print(f"  Path does not exist: {path}")

def extract_tar(model_dir):
    """Extract model.tar.gz and return the extracted directory path."""
    # SageMaker puts the model artifact at either:
    #   <dir>/model.tar.gz          (S3 single-file download)
    #   <dir>/<job-name>/model.tar.gz  (rare, nested)
    tars = glob.glob(os.path.join(model_dir, "**", "model.tar.gz"), recursive=True)
    if not tars:
        raise FileNotFoundError(f"No model.tar.gz found under {model_dir}")
    tar_path = tars[0]
    extract_dir = os.path.dirname(tar_path)
    print(f"  Extracting {tar_path} → {extract_dir}")
    with tarfile.open(tar_path) as t:
        t.extractall(extract_dir)
    print(f"  After extract: {os.listdir(extract_dir)}")
    return extract_dir

def load_xgb(model_dir):
    extract_dir = extract_tar(model_dir)
    # Built-in XGBoost saves as 'xgboost-model'
    candidates = glob.glob(os.path.join(extract_dir, "xgboost-model*"))
    if not candidates:
        raise FileNotFoundError(f"No xgboost-model file found in {extract_dir}")
    model_file = candidates[0]
    print(f"  Loading XGBoost model: {model_file}")
    m = xgb.Booster()
    m.load_model(model_file)
    return m

def load_rf(model_dir):
    extract_dir = extract_tar(model_dir)
    candidates = glob.glob(os.path.join(extract_dir, "*.joblib"))
    if not candidates:
        raise FileNotFoundError(f"No .joblib file found in {extract_dir}")
    model_file = candidates[0]
    print(f"  Loading RF model: {model_file}")
    return joblib.load(model_file)

if __name__ == "__main__":
    try:
        # ── Diagnostics ───────────────────────────────────────
        print("=== EVALUATE MODELS ===")
        list_dir("/opt/ml/processing/model/xgboost")
        list_dir("/opt/ml/processing/model/rf")
        list_dir("/opt/ml/processing/test")

        # ── Load test data ─────────────────────────────────────
        test_files = glob.glob("/opt/ml/processing/test/*.csv")
        if not test_files:
            raise FileNotFoundError("No test CSV found in /opt/ml/processing/test/")
        print(f"  Test file: {test_files[0]}")
        test_df = pd.read_csv(test_files[0], header=None)
        print(f"  Test shape: {test_df.shape}")

        X_te = test_df.iloc[:, 1:].values.astype(float)
        y_te = test_df.iloc[:, 0].values.astype(float)

        results = {}

        # ── XGBoost evaluation ─────────────────────────────────
        try:
            print("\nLoading XGBoost model ...")
            xgb_model = load_xgb("/opt/ml/processing/model/xgboost")
            dtest     = xgb.DMatrix(X_te)
            xgb_prob  = xgb_model.predict(dtest)
            xgb_pred  = (xgb_prob >= 0.5).astype(int)
            results["xgboost"] = {
                "accuracy":  round(float(accuracy_score(y_te, xgb_pred)),  4),
                "auc":       round(float(roc_auc_score(y_te, xgb_prob)),   4),
                "precision": round(float(precision_score(y_te, xgb_pred)), 4),
                "recall":    round(float(recall_score(y_te, xgb_pred)),    4),
                "f1":        round(float(f1_score(y_te, xgb_pred)),        4),
            }
            print(f"  XGBoost metrics: {results['xgboost']}")
        except Exception as e:
            print(f"  XGBoost eval failed: {e}")
            traceback.print_exc()
            results["xgboost"] = {"error": str(e), "accuracy": 0.0}

        # ── Random Forest evaluation ───────────────────────────
        try:
            print("\nLoading Random Forest model ...")
            rf_model = load_rf("/opt/ml/processing/model/rf")
            rf_pred  = rf_model.predict(X_te)
            rf_prob  = rf_model.predict_proba(X_te)[:, 1]
            results["random_forest"] = {
                "accuracy":  round(float(accuracy_score(y_te, rf_pred)),  4),
                "auc":       round(float(roc_auc_score(y_te, rf_prob)),   4),
                "precision": round(float(precision_score(y_te, rf_pred)), 4),
                "recall":    round(float(recall_score(y_te, rf_pred)),    4),
                "f1":        round(float(f1_score(y_te, rf_pred)),        4),
            }
            print(f"  RF metrics: {results['random_forest']}")
        except Exception as e:
            print(f"  RF eval failed: {e}")
            traceback.print_exc()
            results["random_forest"] = {"error": str(e), "accuracy": 0.0}

        # ── Best model ─────────────────────────────────────────
        best = max(results, key=lambda k: results[k].get("accuracy", 0.0))
        results["best_model"] = best
        print(f"\nBest model: {best}")
        print(json.dumps(results, indent=2))

        # ── Write evaluation report ────────────────────────────
        os.makedirs("/opt/ml/processing/evaluation", exist_ok=True)
        out_path = "/opt/ml/processing/evaluation/evaluation.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nEvaluation report written to {out_path}")

    except Exception as e:
        print("=" * 60)
        print(f"EVALUATION FAILED: {e}")
        traceback.print_exc()
        print("=" * 60)
        raise
