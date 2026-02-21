
import argparse, os, glob, traceback, joblib
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def find_csv(directory):
    files = glob.glob(os.path.join(directory, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV found in {directory}. Contents: {os.listdir(directory)}")
    print(f"  Found data file: {files[0]}")
    return files[0]

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--n_estimators", type=int, default=200)
        parser.add_argument("--max_depth",    type=int, default=10)
        parser.add_argument("--model-dir",    type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
        parser.add_argument("--train",        type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
        args = parser.parse_args()

        print(f"Args: {args}")
        print(f"Train dir: {args.train}  contents: {os.listdir(args.train) if args.train else 'None'}")

        data = pd.read_csv(find_csv(args.train), header=None)
        print(f"Data shape: {data.shape}")

        X = data.iloc[:, 1:].values.astype(float)
        y = data.iloc[:, 0].values.astype(float)

        clf = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42, n_jobs=-1
        )
        clf.fit(X, y)

        acc = accuracy_score(y, clf.predict(X))
        print(f"Random Forest → Train Accuracy: {acc:.4f}")

        os.makedirs(args.model_dir, exist_ok=True)
        out_path = os.path.join(args.model_dir, "rf_model.joblib")
        joblib.dump(clf, out_path)
        print(f"Model saved to {out_path}")

    except Exception as e:
        print("=" * 60)
        print(f"TRAINING FAILED: {e}")
        traceback.print_exc()
        print("=" * 60)
        raise
