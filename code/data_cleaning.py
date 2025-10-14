# cleaning.py (excerpt)
import pandas as pd
import numpy as np
from pathlib import Path

LBS_TO_KG = 0.453592
UNIT_MAP = {
    # Extend/verify from data dictionary
    "lb": ("kg", 40, LBS_TO_KG),
    "lbs": ("kg", 40, LBS_TO_KG),
    "kg": ("kg", 40, 1.0),
    "t": ("kg", 40, 1000.0),
}

def parse_dates(df, cols):
    for c in cols:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce", utc=True)
            if s.dt.tz is not None:
                s = s.dt.tz_convert("UTC").dt.tz_localize(None)
            df[c] = s
    return df

def clean_receivals(receivals_df: pd.DataFrame) -> pd.DataFrame:
    df = receivals_df.copy()
    df = parse_dates(df, ["date_arrival"])
    # Drop impossible or un-linkable rows conservatively
    df = df.dropna(subset=["rm_id"])
    df["rm_id"] = df["rm_id"].astype("int64")
    # Ensure numeric net_weight
    df["net_weight"] = pd.to_numeric(df["net_weight"], errors="coerce")
    # Flag invalid and outlier candidates
    df["is_negative_w"] = df["net_weight"] < 0
    df.loc[df["is_negative_w"], "net_weight"] = np.nan  # exclude later if needed
    # Robust outlier flag per rm_id using IQR on log1p scale (flag only)
    tmp = df.groupby("rm_id")["net_weight"].transform(lambda x: np.log1p(x.clip(lower=0)))
    q1 = tmp.groupby(df["rm_id"]).transform("quantile", 0.25)
    q3 = tmp.groupby(df["rm_id"]).transform("quantile", 0.75)
    iqr = q3 - q1
    high = q3 + 3.0 * iqr
    df["is_outlier"] = tmp > high
    # Keep all rows; do not drop outliers here
    return df

def normalize_units(orders_df: pd.DataFrame) -> pd.DataFrame:
    df = orders_df.copy()
    df = parse_dates(df, ["delivery_date", "created_date_time", "modified_date_time"])
    # Normalize quantity and units
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    # Harmonize textual units where available
    if "unit" in df.columns:
        u = df["unit"].str.lower().str.strip()
        conv = u.map({k: v[2] for k, v in UNIT_MAP.items() if isinstance(v, tuple)})
        base = u.map({k: v[0] for k, v in UNIT_MAP.items() if isinstance(v, tuple)})
        df.loc[conv.notna(), "quantity"] = df.loc[conv.notna(), "quantity"] * conv.dropna()
        df.loc[base.notna(), "unit"] = base.dropna()
    # If numeric unit_id codes exist, add rules once verified by dictionary
    df["is_invalid_qty"] = df["quantity"] <= 0
    # Do not impute quantities here; preserve flags for later features
    return df

def main():
    receivals_df = pd.read_csv("data/mod_data/mod_receivals.csv", sep=";")
    orders_df = pd.read_csv("data/mod_data/mod_purchase_orders.csv", sep=";")
    rec_clean = clean_receivals(receivals_df)
    ord_clean = normalize_units(orders_df)
    Path("data_clean").mkdir(exist_ok=True, parents=True)
    rec_clean.to_csv("data_clean/receivals_clean.csv", index=False)
    ord_clean.to_csv("data_clean/purchase_orders_clean.csv", index=False)

if __name__ == "__main__":
    main()