# prediction.py
from __future__ import annotations
import os, json
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from joblib import load

OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", "outputs"))
MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))
PREDICTION_MAPPING_PATH = os.environ.get("PREDICTION_MAPPING_PATH", "prediction_mapping.csv")

def ensure_monotone_by_group(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    df = df.sort_values(["rm_id", "year", "date"])
    df[pred_col] = df.groupby(["rm_id", "year"], group_keys=False)[pred_col] \
                     .transform(lambda s: np.maximum.accumulate(s.values))
    return df

def load_model_and_meta(models_dir: Path) -> tuple[object, dict]:
    meta_path = models_dir / "model_meta.json"
    if not meta_path.exists():
        alt = OUTPUTS_DIR / "model_meta.json"
        if not alt.exists():
            raise FileNotFoundError("model_meta.json not found in models/ or outputs/")
        meta_path = alt
    with open(meta_path, "r") as f:
        meta = json.load(f)

    mode = meta.get("mode", "daily_then_cumulate")
    if mode == "daily_then_cumulate":
        # Find the saved daily model file
        q_used = str(meta.get("q_train_used", "0.50")).replace(".", "")
        model_file = models_dir / f"gbr_daily_q{q_used}.joblib"
        if not model_file.exists():
            # fallback: any daily model
            found = list(models_dir.glob("gbr_daily_q*.joblib"))
            if not found:
                raise FileNotFoundError("Daily model file not found in models/")
            model_file = found[0]
    else:
        # Cumulative model (not the default in this pipeline)
        q_used = str(meta.get("q_train", meta.get("q_train_used", "0.35"))).replace(".", "")
        model_file = models_dir / f"gbr_q{q_used}_cumulative.joblib"
        if not model_file.exists():
            found = list(models_dir.glob("gbr_q*_cumulative.joblib"))
            if not found:
                raise FileNotFoundError("Cumulative model file not found in models/")
            model_file = found[0]

    model = load(model_file)
    return model, meta

def load_prediction_mapping(path: str | Path) -> pd.DataFrame:
    # Try with header, else without
    try:
        pm = pd.read_csv(path)
        # Normalize column names to expected
        lower = {c.lower(): c for c in pm.columns}
        pm = pm.rename(columns={
            lower.get("id", "ID"): "ID",
            lower.get("rm_id", "rm_id"): "rm_id",
            lower.get("forecast_start_date", "forecast_start_date"): "forecast_start_date",
            lower.get("forecast_end_date", "forecast_end_date"): "forecast_end_date",
        })
    except Exception:
        pm = pd.read_csv(path, header=None, names=["ID", "rm_id", "forecast_start_date", "forecast_end_date"])
    pm["rm_id"] = pd.to_numeric(pm["rm_id"], errors="coerce").astype("Int64")
    pm["forecast_start_date"] = pd.to_datetime(pm["forecast_start_date"], errors="coerce")
    pm["forecast_end_date"] = pd.to_datetime(pm["forecast_end_date"], errors="coerce")
    pm = pm.dropna(subset=["ID", "rm_id", "forecast_end_date"]).reset_index(drop=True)
    return pm

def build_calendar_panel_from_mapping(pm: pd.DataFrame) -> pd.DataFrame:
    # Build daily rows from Jan 1, 2025 to May 31, 2025 for all rm_id present in mapping
    rm_ids = pm["rm_id"].dropna().astype("int64").unique().tolist()
    jan1 = pd.Timestamp(year=2025, month=1, day=1)
    may31 = pd.Timestamp(year=2025, month=5, day=31)
    dates = pd.date_range(jan1, may31, freq="D")
    frames = []
    for rid in rm_ids:
        f = pd.DataFrame({"date": dates})
        f["rm_id"] = rid
        f["year"] = 2025
        frames.append(f)
    panel = pd.concat(frames, ignore_index=True)

    # Calendar features only
    dt = panel["date"]
    panel["day_of_year"] = dt.dt.dayofyear.astype("int32")
    panel["week_of_year"] = dt.dt.isocalendar().week.astype("int32")
    panel["month"] = dt.dt.month.astype("int16")
    panel["day_of_month"] = dt.dt.day.astype("int16")
    panel["day_of_week"] = dt.dt.weekday.astype("int8")
    panel["is_weekend"] = panel["day_of_week"].isin([5, 6]).astype("int8")
    panel["is_month_start"] = (dt.dt.day == 1).astype("int8")
    panel["is_month_end"] = (dt.dt.is_month_end).astype("int8")

    return panel.sort_values(["rm_id", "year", "date"]).reset_index(drop=True)

def ensure_feature_columns(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    # Add any missing trained feature columns with zeros, keep order
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
        # coerce numeric
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    # Optionally drop extras not in training features
    keep_cols = set(["rm_id", "year", "date"]) | set(feature_cols)
    cols = [c for c in df.columns if c in keep_cols]
    df = df[["rm_id", "year", "date"] + feature_cols].copy()
    return df

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Load model and metadata
    model, meta = load_model_and_meta(MODELS_DIR)
    mode = meta.get("mode", "daily_then_cumulate")
    feature_cols = meta.get("feature_cols", [])

    # 2) Load prediction mapping
    pm = load_prediction_mapping(PREDICTION_MAPPING_PATH)

    # 3) Build 2025 Jan–May daily panel from mapping and create calendar-only features
    panel = build_calendar_panel_from_mapping(pm)

    # 4) Align features to training feature list (others filled with 0)
    panel = ensure_feature_columns(panel, feature_cols)
    X = panel[feature_cols].values

    # 5) Predict
    out = panel[["rm_id", "year", "date"]].copy()
    if mode == "daily_then_cumulate":
        y_pred_daily = np.clip(model.predict(X), 0.0, None)
        out["y_pred_daily"] = y_pred_daily
        out = out.sort_values(["rm_id", "year", "date"])
        out["y_pred"] = out.groupby(["rm_id", "year"])["y_pred_daily"].cumsum()
    else:
        y_pred = np.clip(model.predict(X), 0.0, None)
        out["y_pred"] = y_pred

    # Enforce monotone cumulative by rm_id-year
    out = ensure_monotone_by_group(out, "y_pred")

    # 6) Merge to mapping by (rm_id, end_date) and build submission
    pred = out.rename(columns={"date": "forecast_end_date"})
    sub = pm.merge(pred[["rm_id", "forecast_end_date", "y_pred"]], on=["rm_id", "forecast_end_date"], how="left")

    # Backward fill within rm_id for any end_date without exact match (should be rare)
    if sub["y_pred"].isna().any():
        left = sub[sub["y_pred"].isna()].copy()
        right = pred.sort_values(["rm_id", "forecast_end_date"])
        filled = pd.merge_asof(
            left.sort_values(["rm_id", "forecast_end_date"]),
            right.sort_values(["rm_id", "forecast_end_date"]),
            on="forecast_end_date",
            by="rm_id",
            direction="backward",
            allow_exact_matches=True,
        )
        sub.loc[sub["y_pred"].isna(), "y_pred"] = filled["y_pred"].values

    sub["y_pred"] = sub["y_pred"].fillna(0.0)
    submission = sub[["ID", "y_pred"]].rename(columns={"y_pred": "predicted_weight"}).copy()
    submission["ID"] = pd.to_numeric(submission["ID"], errors="coerce").astype(int)
    submission = submission.sort_values("ID")

    # 7) Save submission
    out_path = OUTPUTS_DIR / "submission.csv"
    submission.to_csv(out_path, index=False)
    print(f"Saved submission to {out_path} with {len(submission)} rows.")

if __name__ == "__main__":
    main()