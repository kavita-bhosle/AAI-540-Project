
import pandas as pd
import numpy as np
import os, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data",      type=str, default="/opt/ml/processing/input")
    parser.add_argument("--train-output",    type=str, default="/opt/ml/processing/train")
    parser.add_argument("--test-output",     type=str, default="/opt/ml/processing/test")
    parser.add_argument("--baseline-output", type=str, default="/opt/ml/processing/baseline")
    args = parser.parse_args()

    # ── Load ──────────────────────────────────────────────
    csv_file = os.path.join(args.input_data, "Liver_Patient_Dataset__LPD__train.csv")
    df = pd.read_csv(csv_file, encoding="latin-1")
    print(f"Loaded {len(df)} rows, {df.shape[1]} cols")

    # ── Strip invisible chars from column names (\xa0 prefix) ──
    df.columns = df.columns.str.strip().str.replace("\xa0", "", regex=False)

    # ── Rename columns to clean short names ───────────────
    df.rename(columns={
        "Age of the patient":                  "Age",
        "Gender of the patient":               "Gender",
        "Total Bilirubin":                     "Total_Bilirubin",
        "Direct Bilirubin":                    "Direct_Bilirubin",
        "Alkphos Alkaline Phosphotase":        "Alkaline_Phosphotase",
        "Sgpt Alamine Aminotransferase":       "Alamine_Aminotransferase",
        "Sgot Aspartate Aminotransferase":     "Aspartate_Aminotransferase",
        "Total Protiens":                      "Total_Proteins",
        "ALB Albumin":                         "Albumin",
        "A/G Ratio Albumin and Globulin Ratio": "AG_Ratio",
        "Result":                              "Target"
    }, inplace=True)

    # ── Clean ─────────────────────────────────────────────
    df.dropna(inplace=True)
    df["Gender"] = (df["Gender"] == "Male").astype(int)
    # Target: 1 = liver disease, 0 = no disease  (original: 1=disease, 2=no disease)
    df["Target"] = (df["Target"] == 1).astype(int)

    # ── Feature Engineering ───────────────────────────────
    df["Bilirubin_ratio"] = df["Direct_Bilirubin"] / (df["Total_Bilirubin"] + 1e-6)
    df["Enzyme_ratio"]    = (df["Alamine_Aminotransferase"] /
                             (df["Aspartate_Aminotransferase"] + 1e-6))
    df["Age_group"]       = pd.cut(df["Age"],
                                   bins=[0, 30, 50, 70, 120],
                                   labels=[0, 1, 2, 3]).astype(int)

    print(f"Class distribution: {df['Target'].value_counts().to_dict()}")

    # ── Split ─────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    feature_cols = [c for c in df.columns if c != "Target"]
    X = df[feature_cols]
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_test_sc  = pd.DataFrame(scaler.transform(X_test),      columns=feature_cols)

    # ── Save ──────────────────────────────────────────────
    os.makedirs(args.train_output,    exist_ok=True)
    os.makedirs(args.test_output,     exist_ok=True)
    os.makedirs(args.baseline_output, exist_ok=True)

    train_df = pd.concat([y_train.reset_index(drop=True),
                          X_train_sc.reset_index(drop=True)], axis=1)
    test_df  = pd.concat([y_test.reset_index(drop=True),
                          X_test_sc.reset_index(drop=True)], axis=1)

    train_df.to_csv(os.path.join(args.train_output,    "train.csv"),    index=False, header=False)
    test_df.to_csv(os.path.join(args.test_output,      "test.csv"),     index=False, header=False)
    test_df.to_csv(os.path.join(args.baseline_output,  "baseline.csv"), index=False)   # with header for Monitor

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    print("Preprocessing complete.")
