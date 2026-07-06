"""
baselines.py
------------
Random Forest, XGBoost, and Neural Network baselines used for the SOTA
comparison against the GMDH cost/demand models (Fig. 5a / results table).

Evaluated with the same stratified 5-fold cross-validation protocol as
R/cv_pipeline.R, over data/energy_balance_2019_2023.csv.

Environment: Python 3.11, scikit-learn 1.4, xgboost 2.0, torch 2.2
Seed = 42
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

SEED = 42
np.random.seed(SEED)

FEATURES = ["E_SPP", "E_WPP", "E_HPP", "E_BCP", "E_CPSS"]
TARGETS = {"demand": "D", "cost": "C"}


def cross_validate(model_factory, X, y, k=5, seed=SEED, scale=False):
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    rmses, maes, r2s = [], [], []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if scale:
            scaler = StandardScaler().fit(X_tr)
            X_tr = scaler.transform(X_tr)
            X_te = scaler.transform(X_te)

        model = model_factory()
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        rmses.append(np.sqrt(mean_squared_error(y_te, y_pred)))
        maes.append(mean_absolute_error(y_te, y_pred))
        r2s.append(r2_score(y_te, y_pred))

    return {
        "rmse_mean": np.mean(rmses), "rmse_std": np.std(rmses),
        "mae_mean": np.mean(maes), "mae_std": np.std(maes),
        "r2_mean": np.mean(r2s), "r2_std": np.std(r2s),
    }


def run_all_baselines(df: pd.DataFrame, target_col: str):
    X = df[FEATURES].values
    y = df[target_col].values
    # normalize the target the same way the manuscript reports (min-max to [0,1])
    y_norm = (y - y.min()) / (y.max() - y.min())

    results = {}

    results["RandomForest"] = cross_validate(
        lambda: RandomForestRegressor(
            n_estimators=300, max_depth=6, random_state=SEED
        ),
        X, y_norm,
    )

    if HAS_XGB:
        results["XGBoost"] = cross_validate(
            lambda: XGBRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                verbosity=0,
            ),
            X, y_norm,
        )
    else:
        print("xgboost not installed; skipping XGBoost baseline")

    results["NeuralNetwork"] = cross_validate(
        lambda: MLPRegressor(
            hidden_layer_sizes=(32, 16), activation="relu",
            max_iter=2000, random_state=SEED,
        ),
        X, y_norm, scale=True,
    )

    return results


def main():
    df = pd.read_csv("data/energy_balance_2019_2023.csv")

    all_rows = []
    for label, col in TARGETS.items():
        print(f"\n=== Baseline comparison: {label} model (target = {col}) ===")
        res = run_all_baselines(df, col)
        for model_name, m in res.items():
            print(f"{model_name:15s} RMSE={m['rmse_mean']:.4f}±{m['rmse_std']:.4f}  "
                  f"MAE={m['mae_mean']:.4f}±{m['mae_std']:.4f}  "
                  f"R2={m['r2_mean']:.4f}±{m['r2_std']:.4f}")
            all_rows.append({"target": label, "model": model_name, **m})

    out = pd.DataFrame(all_rows)
    out.to_csv("results/baseline_metrics.csv", index=False)
    print("\nSaved results/baseline_metrics.csv")


if __name__ == "__main__":
    main()
