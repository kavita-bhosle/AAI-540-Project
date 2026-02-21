
import argparse, os, glob, traceback
import pandas as pd, numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

def find_csv(directory):
    """Find the first CSV in a directory regardless of filename."""
    files = glob.glob(os.path.join(directory, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV found in {directory}. Contents: {os.listdir(directory)}")
    print(f"  Found data file: {files[0]}")
    return files[0]

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--max_depth",   type=int,   default=5)
        parser.add_argument("--eta",         type=float, default=0.2)
        parser.add_argument("--subsample",   type=float, default=0.8)
        parser.add_argument("--num_round",   type=int,   default=100)
        parser.add_argument("--objective",   type=str,   default="binary:logistic")
        parser.add_argument("--eval_metric", type=str,   default="auc")
        parser.add_argument("--model-dir",   type=str,   default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
        parser.add_argument("--train",       type=str,   default=os.environ.get("SM_CHANNEL_TRAIN"))
        parser.add_argument("--test",        type=str,   default=os.environ.get("SM_CHANNEL_TEST"))
        args = parser.parse_args()

        print(f"Args: {args}")
        print(f"Train dir: {args.train}  contents: {os.listdir(args.train) if args.train else 'None'}")
        if args.test:
            print(f"Test  dir: {args.test}   contents: {os.listdir(args.test)}")

        # ── Load training data ────────────────────────────────
        train = pd.read_csv(find_csv(args.train), header=None)
        print(f"Train shape: {train.shape}")

        X_tr = train.iloc[:, 1:].values.astype(float)
        y_tr = train.iloc[:, 0].values.astype(float)
        dtrain = xgb.DMatrix(X_tr, label=y_tr)

        # ── Load test data (optional – used for early stopping) ──
        evals = [(dtrain, "train")]
        if args.test and os.path.isdir(args.test):
            test  = pd.read_csv(find_csv(args.test), header=None)
            print(f"Test shape: {test.shape}")
            X_te  = test.iloc[:, 1:].values.astype(float)
            y_te  = test.iloc[:, 0].values.astype(float)
            dtest = xgb.DMatrix(X_te, label=y_te)
            evals.append((dtest, "test"))
        else:
            print("No test channel found — training without early stopping.")
            dtest = None

        # ── Train ─────────────────────────────────────────────
        params = {
            "max_depth":   args.max_depth,
            "eta":         args.eta,
            "subsample":   args.subsample,
            "objective":   args.objective,
            "eval_metric": args.eval_metric,
            "seed":        42,
        }
        print(f"Training params: {params}")

        callbacks = []
        if dtest is not None:
            callbacks = [xgb.callback.EarlyStopping(rounds=10)]

        model = xgb.train(
            params, dtrain, args.num_round,
            evals=evals,
            callbacks=callbacks,
            verbose_eval=10
        )

        # ── Evaluate ──────────────────────────────────────────
        if dtest is not None:
            preds = (model.predict(dtest) >= 0.5).astype(int)
            acc   = accuracy_score(y_te, preds)
            auc   = roc_auc_score(y_te, model.predict(dtest))
            print(f"XGBoost → Accuracy: {acc:.4f}  AUC: {auc:.4f}")

        # ── Save ──────────────────────────────────────────────
        os.makedirs(args.model_dir, exist_ok=True)
        out_path = os.path.join(args.model_dir, "xgboost-model")
        model.save_model(out_path)
        print(f"Model saved to {out_path}")

    except Exception as e:
        print("=" * 60)
        print(f"TRAINING FAILED: {e}")
        traceback.print_exc()
        print("=" * 60)
        raise   # re-raise so SageMaker marks the job as failed
