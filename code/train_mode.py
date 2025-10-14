# train_model.py
from __future__ import annotations
import os, json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump

# Optional Azure ML imports
try:
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    AZURE_AVAILABLE = True
except Exception:
    AZURE_AVAILABLE = False

# --------------------
# Config (env vars)
# --------------------
USE_AZURE_ASSET = os.environ.get("USE_AZURE_ASSET", "1") == "1"
ASSET_NAME = os.environ.get("ASSET_NAME", "model_ready_data")
ASSET_VERSION = os.environ.get("ASSET_VERSION", "5")
LOCAL_DATASET_PATH = os.environ.get("LOCAL_DATASET_PATH", "data_model/panel_jan_may_features.csv")

# Daily-to-cumulative mode enforced
Q_TRAIN_PRIMARY = float(os.environ.get("Q_TRAIN_PRIMARY", "0.50"))  # start at 0.5 to escape zero plateau
Q_TRAIN_FALLBACK = float(os.environ.get("Q_TRAIN_FALLBACK", "0.35"))  # fallback if needed
Q_EVAL = float(os.environ.get("Q_EVAL", "0.20"))  # report conservative score

NONZERO_WEIGHT = float(os.environ.get("NONZERO_WEIGHT", "8.0"))  # upweight nonzero daily targets

N_ESTIMATORS = int(os.environ.get("N_ESTIMATORS", "1500"))  # fewer trees for speed; increase if needed
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "0.03"))
MAX_DEPTH = int(os.environ.get("MAX_DEPTH", "5"))
SUBSAMPLE = float(os.environ.get("SUBSAMPLE", "0.9"))
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "42"))

OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", "outputs"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))

# --------------------
# Utilities
# --------------------
def pinball(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.where(diff >= 0, q * diff, (1.0 - q) * (-diff))))

def ensure_monotone_by_group(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    df = df.sort_values(["rm_id", "year", "date"])
    df[pred_col] = df.groupby(["rm_id", "year"], group_keys=False)[pred_col] \
                     .transform(lambda s: np.maximum.accumulate(s.values))
    return df

def load_dataset_local(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["rm_id", "year", "date"]).reset_index(drop=True)

def load_dataset_azure(asset_name: str, asset_version: str) -> pd.DataFrame:
    if not AZURE_AVAILABLE:
        raise RuntimeError("Azure ML libraries not available.")
    ml_client = MLClient.from_config(credential=DefaultAzureCredential())
    data_asset = ml_client.data.get(name=asset_name, version=asset_version)
    df = pd.read_csv(data_asset.path, parse_dates=["date"])
    return df.sort_values(["rm_id", "year", "date"]).reset_index(drop=True)

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    drop_cols = {"rm_id", "year", "date", "daily_net_weight", "cum_net_weight"}
    feats = [c for c in df.columns if c not in drop_cols]
    feats = [c for c in feats if np.issubdtype(df[c].dtype, np.number)]
    return feats

def split_by_year(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[int], int]:
    years = sorted(df["year"].unique().tolist())
    if len(years) < 2:
        raise ValueError("Need at least two years for temporal validation.")
    train_years, valid_year = years[:-1], years[-1]
    return df[df["year"].isin(train_years)].copy(), df[df["year"] == valid_year].copy(), train_years, valid_year

def make_gbr(alpha: float) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        subsample=SUBSAMPLE,
        random_state=RANDOM_STATE,
    )

def train_predict_daily_then_cumulate(
    train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: List[str], alpha: float
) -> Tuple[pd.DataFrame, float, GradientBoostingRegressor]:
    # Train on daily targets
    y_train = train_df["daily_net_weight"].values
    y_valid = valid_df["daily_net_weight"].values
    X_train = train_df[feature_cols].values
    X_valid = valid_df[feature_cols].values

    # Upweight nonzero daily observations
    sw_train = np.where(y_train > 0.0, NONZERO_WEIGHT, 1.0)

    model = make_gbr(alpha=alpha)
    model.fit(X_train, y_train, sample_weight=sw_train)

    # Predict daily and cumulate per rm_id-year
    valid = valid_df.copy()
    valid["y_pred_raw_daily"] = np.clip(model.predict(X_valid), 0.0, None)
    valid = valid.sort_values(["rm_id", "year", "date"])
    valid["y_pred_raw"] = valid.groupby(["rm_id", "year"])["y_pred_raw_daily"].cumsum()

    # Enforce monotone cumulative curve
    valid = ensure_monotone_by_group(valid, "y_pred_raw")
    valid = valid.rename(columns={"y_pred_raw": "y_pred"})

    # Evaluate on cumulative target at q=Q_EVAL
    val_loss = pinball(valid["cum_net_weight"].values, valid["y_pred"].values, q=Q_EVAL)
    return valid[["rm_id", "year", "date", "cum_net_weight", "y_pred"]], val_loss, model

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    if USE_AZURE_ASSET:
        try:
            df = load_dataset_azure(ASSET_NAME, ASSET_VERSION)
        except Exception as e:
            print(f"Azure load error: {e} — falling back to local dataset")
            df = load_dataset_local(LOCAL_DATASET_PATH)
    else:
        df = load_dataset_local(LOCAL_DATASET_PATH)

    # 2) Features and split
    feature_cols = get_feature_columns(df)
    # Ensure numeric types (Azure CSVs can coerce to object if NaNs appear)
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Daily target must exist
    if "daily_net_weight" not in df.columns or "cum_net_weight" not in df.columns:
        raise ValueError("Dataset must include daily_net_weight and cum_net_weight columns")

    train_df, valid_df, train_years, valid_year = split_by_year(df)

    # Diagnostics
    frac_pos_daily = float(np.mean(train_df["daily_net_weight"].values > 0.0))
    print(f"Train rows: {len(train_df)}, Valid rows: {len(valid_df)}, n_features: {len(feature_cols)}")
    print(f"Fraction of positive daily targets in train: {frac_pos_daily:.3f}")

    # 3) Primary attempt at higher quantile (more likely to move off zero)
    valid_primary, val_loss_primary, model_primary = train_predict_daily_then_cumulate(
        train_df, valid_df, feature_cols, alpha=Q_TRAIN_PRIMARY
    )
    all_zero_primary = bool(np.allclose(valid_primary["y_pred"].values, 0.0))
    print(f"[Primary q={Q_TRAIN_PRIMARY}] Pinball(q={Q_EVAL}) on {valid_year}: {val_loss_primary:.6f} | all_zero={all_zero_primary}")

    # 4) Fallback attempt at a slightly lower quantile if predictions collapsed
    if all_zero_primary:
        print(f"Primary predictions are all zero; retrying with q={Q_TRAIN_FALLBACK} and same settings...")
        valid_fallback, val_loss_fallback, model_fallback = train_predict_daily_then_cumulate(
            train_df, valid_df, feature_cols, alpha=Q_TRAIN_FALLBACK
        )
        chosen = (valid_fallback, val_loss_fallback, model_fallback, Q_TRAIN_FALLBACK)
    else:
        chosen = (valid_primary, val_loss_primary, model_primary, Q_TRAIN_PRIMARY)

    valid_out, val_loss, model, q_used = chosen

    # 5) Save artifacts
    tag = f"daily_q{q_used}".replace(".", "")
    dump(model, MODELS_DIR / f"gbr_{tag}.joblib")
    meta = {
        "mode": "daily_then_cumulate",
        "q_train_used": q_used,
        "q_eval": Q_EVAL,
        "nonzero_weight": NONZERO_WEIGHT,
        "n_estimators": N_ESTIMATORS,
        "learning_rate": LEARNING_RATE,
        "max_depth": MAX_DEPTH,
        "subsample": SUBSAMPLE,
        "valid_year": valid_year,
        "train_years": train_years,
        "feature_cols": feature_cols,
        "data_source": "azure_asset" if USE_AZURE_ASSET else "local_csv",
        "asset_name": ASSET_NAME if USE_AZURE_ASSET else "",
        "asset_version": ASSET_VERSION if USE_AZURE_ASSET else "",
    }
    with open(MODELS_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(OUTPUTS_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    valid_out.to_csv(OUTPUTS_DIR / f"valid_preds_{valid_year}.csv", index=False)
    print(f"[Chosen q={q_used}] Validation pinball(q={Q_EVAL}) on {valid_year}: {val_loss:.6f}")
    print(f"Saved model to {MODELS_DIR}/gbr_{tag}.joblib and predictions to {OUTPUTS_DIR}/valid_preds_{valid_year}.csv")

if __name__ == "__main__":
    main()