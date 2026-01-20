import argparse
import os
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def main(x_train_path: str, x_test_path: str, out_dir: str, method: str, scaler_path: str):
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)

    # ✅ GARDE UNIQUEMENT LES COLONNES NUMÉRIQUES
    X_train = X_train.select_dtypes(include=["number"])
    X_test = X_test[X_train.columns]  # même ordre / mêmes colonnes

    if method == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(
        f"{out_dir}/X_train_scaled.csv", index=False
    )
    pd.DataFrame(X_test_scaled, columns=X_train.columns).to_csv(
        f"{out_dir}/X_test_scaled.csv", index=False
    )

    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    dump(scaler, scaler_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--x_train", required=True)
    p.add_argument("--x_test", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--method", choices=["standard", "minmax"], default="standard")
    p.add_argument("--scaler_path", default="models/data/scaler.pkl")
    args = p.parse_args()

    main(args.x_train, args.x_test, args.out_dir, args.method, args.scaler_path)
