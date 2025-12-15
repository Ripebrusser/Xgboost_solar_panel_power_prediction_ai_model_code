import os
import json
import time
import joblib
import warnings
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from xgboost import XGBRegressor, DMatrix
import xgboost as xgb

warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

# -----------------------------
# CONFIGURATION
# -----------------------------
CONFIG = {
    "weather_csv": "Plant_1_Weather_Sensor_Data.csv",
    "generation_csv": "Plant_1_Generation_Data.csv",
    "output_dir": "xgb_output",
    "n_lags": 24,                     # number of lag steps (15-min intervals, 24=6 hours)
    "rolling_windows": [3, 6, 12, 24],
    "test_fraction": 0.15,            # holdout test fraction (chronological)
    "val_fraction": 0.15,             # validation fraction taken from end of train
    "random_state": 42,
    "scale_features": True,
    "xgb_defaults": {
        "objective": "reg:squarederror",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 0,
        "n_jobs": -1,
        "random_state": 42
    },
    "random_search": {
        "n_iter": 50,
        "cv_splits": 3,
        "scoring": "neg_root_mean_squared_error",
    },
    "early_stopping_rounds": 50,
    "plots_window": 500,               # points to show in time-series plots
    "shap": True,                      # compute SHAP summary if shap installed
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)

# -----------------------------
# Utilities
# -----------------------------
def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# -----------------------------
# Custom Transformers (sklearn-style)
# -----------------------------
class TimeFeatures(BaseEstimator, TransformerMixin):
    """Add cyclical time features and simple calendar features"""
    def __init__(self, datetime_col: str = "DATE_TIME"):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        dt = pd.to_datetime(X[self.datetime_col])
        X["hour"] = dt.dt.hour
        X["minute"] = dt.dt.minute
        X["dayofyear"] = dt.dt.dayofyear
        X["weekday"] = dt.dt.weekday
        # cyclical encoding
        X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24.0)
        X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24.0)
        return X

class LagRollingFeatures(BaseEstimator, TransformerMixin):
    """Create lags and rolling statistics for a target column"""
    def __init__(self, target_col: str = "AC_POWER", n_lags: int = 24, rolling_windows: List[int] = [3,6,12]):
        self.target_col = target_col
        self.n_lags = n_lags
        self.rolling_windows = rolling_windows

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy().sort_values("DATE_TIME").reset_index(drop=True)
        for lag in range(1, self.n_lags + 1):
            X[f"lag_{lag}"] = X[self.target_col].shift(lag)
        for w in self.rolling_windows:
            X[f"roll_mean_{w}"] = X[self.target_col].shift(1).rolling(window=w, min_periods=1).mean()
            X[f"roll_std_{w}"] = X[self.target_col].shift(1).rolling(window=w, min_periods=1).std().fillna(0)
        return X

# -----------------------------
# Loading and merging data
# -----------------------------
def load_and_merge(weather_path: str, generation_path: str) -> pd.DataFrame:
    """Load CSVs, parse datetimes, aggregate generation per timestamp and merge with weather."""
    weather = pd.read_csv(weather_path)
    gen = pd.read_csv(generation_path)

    # Parse timestamps
    weather["DATE_TIME"] = pd.to_datetime(weather["DATE_TIME"], errors="coerce", infer_datetime_format=True)
    gen["DATE_TIME"] = pd.to_datetime(gen["DATE_TIME"], errors="coerce", infer_datetime_format=True)

    # drop rows missing timestamp
    weather = weather.dropna(subset=["DATE_TIME"]).reset_index(drop=True)
    gen = gen.dropna(subset=["DATE_TIME"]).reset_index(drop=True)

    # Aggregate generation to plant-level (sum AC_POWER per timestamp)
    if "AC_POWER" not in gen.columns:
        raise ValueError("Generation data must contain 'AC_POWER' column.")
    gen_agg = gen.groupby("DATE_TIME", as_index=False).agg({"AC_POWER": "sum"})

    # Merge on DATE_TIME (inner join ensures matched timestamps)
    df = pd.merge(weather, gen_agg, on="DATE_TIME", how="inner").sort_values("DATE_TIME").reset_index(drop=True)
    return df

# -----------------------------
# Preprocessing pipeline
# -----------------------------
def preprocess(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Full preprocessing pipeline: type coercion, remove invalids, features."""
    df = df.copy()
    # Numeric coercion
    for col in ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "AC_POWER"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Drop rows with missing critical values
    df = df.dropna(subset=["DATE_TIME", "IRRADIATION", "AC_POWER"]).reset_index(drop=True)

    # Add time features
    tf = TimeFeatures()
    df = tf.transform(df)

    # Add lags and rolling
    lag = LagRollingFeatures(target_col="AC_POWER", n_lags=cfg["n_lags"], rolling_windows=cfg["rolling_windows"])
    df = lag.transform(df)

    # Drop rows with NaNs introduced by lags
    df = df.dropna().reset_index(drop=True)

    return df

# -----------------------------
# Train/validation/test split (chronological)
# -----------------------------
def chrono_split(df: pd.DataFrame, cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns train_df, val_df, test_df according to chronological splits.
    Test set is last test_fraction of data.
    Validation fraction is taken from the end of the remaining train set.
    """
    n = len(df)
    test_n = int(np.floor(cfg["test_fraction"] * n))
    test_df = df.iloc[-test_n:].reset_index(drop=True)
    trainval_df = df.iloc[:-test_n].reset_index(drop=True)

    # Validation from end of trainval
    val_n = int(np.floor(cfg["val_fraction"] * len(trainval_df)))
    val_df = trainval_df.iloc[-val_n:].reset_index(drop=True)
    train_df = trainval_df.iloc[:-val_n].reset_index(drop=True)

    return train_df, val_df, test_df

# -----------------------------
# Feature selection helper
# -----------------------------
def get_feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"DATE_TIME", "PLANT_ID", "SOURCE_KEY", "AC_POWER"}
    return [c for c in df.columns if c not in exclude]

# -----------------------------
# Evaluation metrics
# -----------------------------
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(1e-6, np.abs(y_true)))) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Bias": bias, "MAPE(%)": mape}

# -----------------------------
# Hyperparameter search (TimeSeriesSplit + RandomizedSearchCV)
# -----------------------------
def hyperparameter_search(X: pd.DataFrame, y: pd.Series, cfg: Dict):
    param_dist = {
        "n_estimators": [200, 400, 600, 1000],
        "max_depth": [3, 4, 6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0, 1e-2, 1e-1, 1],
        "reg_lambda": [0.5, 1, 1.5, 2]
    }
    base = XGBRegressor(**cfg["xgb_defaults"])
    tscv = TimeSeriesSplit(n_splits=cfg["random_search"]["cv_splits"])
    rsearch = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=cfg["random_search"]["n_iter"],
        scoring=cfg["random_search"]["scoring"],
        cv=tscv,
        verbose=2,
        random_state=cfg["random_state"],
        n_jobs=-1
    )
    rsearch.fit(X, y)
    return rsearch.best_params_, rsearch.best_score_, rsearch

# -----------------------------
# Final training with early stopping (using validation set)
# -----------------------------
def train_final_model(X_train, y_train, X_val, y_val, best_params: Dict, cfg: Dict):
    params = cfg["xgb_defaults"].copy()
    params.update(best_params)
    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=cfg["early_stopping_rounds"],
        verbose=False
    )
    return model

# -----------------------------
# Plotting helpers
# -----------------------------
def plot_actual_vs_pred(time_index, y_true, y_pred, outpath, title="Actual vs Predicted", window=None):
    window = window or len(y_true)
    plt.figure(figsize=(14,6))
    plt.plot(time_index[:window], y_true[:window], label="Actual", color="black", linewidth=2)
    plt.plot(time_index[:window], y_pred[:window], label="Predicted", alpha=0.9)
    plt.xlabel("Time")
    plt.ylabel("AC Power (kW)")
    plt.xticks(rotation=45)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_error_hist(y_true, preds: Dict[str, np.ndarray], outpath):
    plt.figure(figsize=(10,6))
    for name, pred in preds.items():
        err = y_true - pred
        sns.kdeplot(err, label=name)
    plt.axvline(0, color="k", linestyle="--", linewidth=1)
    plt.xlabel("Prediction error (kW)")
    plt.title("Error Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_feature_importance(model, feature_names, outpath, top_n=20):
    fi = model.feature_importances_
    idx = np.argsort(fi)[-top_n:]
    plt.figure(figsize=(8, max(4, top_n*0.3)))
    plt.barh(np.array(feature_names)[idx], fi[idx])
    plt.title("XGBoost Feature Importance (gain)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

# -----------------------------
# SHAP explainability (optional)
# -----------------------------
def compute_shap(model, X_sample, outdir):
    try:
        import shap
    except Exception:
        print("SHAP not installed. Skipping SHAP analysis.")
        return
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    # summary
    plt.figure(figsize=(8,6))
    shap.plots.beeswarm(shap_values, show=False)
    plt.title("SHAP summary (beeswarm)")
    plt.tight_layout()
    path = os.path.join(outdir, "shap_beeswarm.png")
    plt.savefig(path, dpi=300)
    plt.close()
    # feature importance (mean abs SHAP)
    shap.summary_plot(shap_values, X_sample, show=False, plot_size=(8,6))
    plt.tight_layout()
    path2 = os.path.join(outdir, "shap_summary.png")
    plt.savefig(path2, dpi=300)
    plt.close()

# -----------------------------
# Main pipeline
# -----------------------------
def run_pipeline(cfg=CONFIG):
    ts = now_stamp()
    run_dir = os.path.join(cfg["output_dir"], f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    # 1. Load & merge
    print("Loading and merging data...")
    df = load_and_merge(cfg["weather_csv"], cfg["generation_csv"])
    print(f"Loaded {len(df)} rows after merge.")

    # 2. Preprocess & feature engineering
    print("Preprocessing and feature engineering...")
    df_processed = preprocess(df, cfg)
    print(f"After preprocessing: {len(df_processed)} rows.")

    # 3. Split
    train_df, val_df, test_df = chrono_split(df_processed, cfg)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    feature_cols = get_feature_columns(df_processed)
    print(f"Number of features: {len(feature_cols)}")

    X_train = train_df[feature_cols]
    y_train = train_df["AC_POWER"]
    X_val = val_df[feature_cols]
    y_val = val_df["AC_POWER"]
    X_test = test_df[feature_cols]
    y_test = test_df["AC_POWER"]

    # 4. Scaling (if enabled)
    scaler = None
    if cfg["scale_features"]:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
        X_val = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)
        joblib.dump(scaler, os.path.join(run_dir, "scaler.joblib"))
        print("Saved scaler.")

    # Save a quick snapshot of feature stats
    X_train.describe().to_csv(os.path.join(run_dir, "feature_stats_train.csv"))

    # 5. Hyperparameter search (on training set only)
    print("Starting hyperparameter search (this may take time)...")
    best_params, best_score, _ = hyperparameter_search(X_train, y_train, cfg)
    print("Best params found:", best_params)
    save_json({"best_params": best_params, "best_cv_score": best_score}, os.path.join(run_dir, "hyperopt_result.json"))

    # 6. Final training with early stopping (use validation set)
    print("Training final model with early stopping on validation set...")
    model = train_final_model(X_train, y_train, X_val, y_val, best_params, cfg)
    # Save model (joblib and xgb native)
    joblib.dump(model, os.path.join(run_dir, "xgb_model.joblib"))
    model.get_booster().save_model(os.path.join(run_dir, "xgb_model.json"))
    print("Saved final model.")

    # 7. Predict & evaluate on test set
    print("Predicting on test set...")
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test.values, y_pred)
    print("Test metrics:", metrics)
    save_json(metrics, os.path.join(run_dir, "test_metrics.json"))

    # 8. Diagnostic plots
    time_index = test_df["DATE_TIME"].reset_index(drop=True)
    plot_actual_vs_pred(time_index, y_test.values, y_pred, os.path.join(run_dir, "actual_vs_pred.png"), window=cfg["plots_window"])
    plot_feature_importance(model, feature_cols, os.path.join(run_dir, "feature_importance.png"), top_n=25)
    plot_error_hist(y_test.values, {"XGBoost": y_pred}, os.path.join(run_dir, "error_distribution.png"))

    # 9. SHAP analysis (optional, sample)
    if cfg.get("shap", False):
        print("Computing SHAP explanations (may be slow)...")
        # sample a subset from train for SHAP
        X_sample = X_train.sample(n=min(2000, len(X_train)), random_state=cfg["random_state"])
        compute_shap(model, X_sample, run_dir)

    # 10. Save predictions (for downstream analysis)
    out_df = test_df[["DATE_TIME"]].copy()
    out_df["actual"] = y_test.values
    out_df["predicted"] = y_pred
    out_df.to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False)
    print(f"Run artifacts saved to {run_dir}")

    return run_dir, metrics

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    run_dir, metrics = run_pipeline(CONFIG)
    print("Done. Artifacts in:", run_dir)
    print("Metrics:", metrics)
