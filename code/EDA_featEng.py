# EdaFeatModeling.py
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

JAN, MAY = 1, 5
LOOKBACK_YEARS = 4

def load_clean(data_dir: str | Path = "data_clean"):
    data_dir = Path(data_dir)
    rec = pd.read_csv(data_dir / "receivals_clean.csv", low_memory=False)

    # Parse and normalize dates
    rec["date_arrival"] = pd.to_datetime(rec["date_arrival"], errors="coerce")
    rec["date"] = rec["date_arrival"].dt.floor("D")

    # Ensure numeric
    rec["net_weight"] = pd.to_numeric(rec["net_weight"], errors="coerce").fillna(0.0)

    # Keep essential columns
    keep = ["rm_id", "date", "net_weight"]
    rec = rec[keep].dropna(subset=["rm_id", "date"])
    rec["rm_id"] = rec["rm_id"].astype("int64")
    return rec

def build_full_daily(rec: pd.DataFrame) -> pd.DataFrame:
    # Aggregate receivals to daily flows per rm_id
    daily = (
        rec.groupby(["rm_id", "date"], as_index=False)["net_weight"]
           .sum()
           .rename(columns={"net_weight": "daily_net_weight"})
    )

    # Reindex to continuous daily calendar per rm_id for robust rolling features
    frames = []
    for rm_id, g in daily.groupby("rm_id", as_index=False):
        g = g.sort_values("date")
        start = g["date"].min()
        end = g["date"].max()
        idx = pd.date_range(start, end, freq="D")
        f = pd.DataFrame({"date": idx})
        f["rm_id"] = rm_id
        f = f.merge(g, on=["rm_id", "date"], how="left")
        f["daily_net_weight"] = f["daily_net_weight"].fillna(0.0)
        frames.append(f)

    full_daily = pd.concat(frames, ignore_index=True)
    full_daily["year"] = full_daily["date"].dt.year
    return full_daily

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["date"]
    df["day_of_year"] = dt.dt.dayofyear.astype("int32")
    df["week_of_year"] = dt.dt.isocalendar().week.astype("int32")
    df["month"] = dt.dt.month.astype("int16")
    df["day_of_month"] = dt.dt.day.astype("int16")
    df["day_of_week"] = dt.dt.weekday.astype("int8")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")
    df["is_month_start"] = (dt.dt.day == 1).astype("int8")
    df["is_month_end"] = (dt.dt.is_month_end).astype("int8")
    return df

def add_rolling_cadence_features(full_daily: pd.DataFrame) -> pd.DataFrame:
    full_daily = full_daily.sort_values(["rm_id", "date"])

    # Rolling sums and counts over past windows (shifted to avoid same-day leakage)
    wins = [7, 28, 56, 112, 224, 365]
    def _roll_sum(s: pd.Series, w: int) -> pd.Series:
        return s.rolling(w, min_periods=1).sum().shift(1)

    def _roll_cnt_pos(s: pd.Series, w: int) -> pd.Series:
        return (s > 0).astype("int8").rolling(w, min_periods=1).sum().shift(1)

    for w in wins:
        full_daily[f"roll_sum_{w}"] = (
            full_daily.groupby("rm_id", group_keys=False)["daily_net_weight"]
                      .transform(lambda s: _roll_sum(s, w))
                      .fillna(0.0)
                      .astype("float64")
        )
        full_daily[f"roll_cnt_{w}"] = (
            full_daily.groupby("rm_id", group_keys=False)["daily_net_weight"]
                      .transform(lambda s: _roll_cnt_pos(s, w))
                      .fillna(0.0)
                      .astype("float64")
        )

    # Days since last nonzero receival (no groupby.apply to avoid warnings)
    flow = full_daily["daily_net_weight"] > 0
    flow_date = full_daily["date"].where(flow)
    last_flow_date = flow_date.groupby(full_daily["rm_id"]).ffill()
    full_daily["days_since_last"] = (
        (full_daily["date"] - last_flow_date).dt.days.fillna(999).astype("int32")
    )
    return full_daily

def build_supervised_panel(full_daily: pd.DataFrame) -> pd.DataFrame:
    # Keep Jan–May rows only and compute cumulative target
    mask = (full_daily["date"].dt.month >= JAN) & (full_daily["date"].dt.month <= MAY)
    jm = full_daily.loc[mask].copy()
    jm = jm.sort_values(["rm_id", "year", "date"])
    jm["cum_net_weight"] = (
        jm.groupby(["rm_id", "year"])["daily_net_weight"].cumsum()
    )
    return jm

def add_prior_year_features(jm: pd.DataFrame) -> pd.DataFrame:
    jm = jm.copy()
    # Month-day key for same-calendar-day grouping
    jm["md"] = (jm["date"].dt.month * 100 + jm["date"].dt.day).astype("int16")

    # Prior-year same-day cumulative (previous observation in (rm_id, md) order)
    jm = jm.sort_values(["rm_id", "md", "year", "date"])
    jm["prev_year_cum"] = (
        jm.groupby(["rm_id", "md"])["cum_net_weight"].shift(1)
    )

    # Expanding median up to previous observation, aligned via transform to avoid index mismatch
    jm["median_cum_same_day"] = (
        jm.groupby(["rm_id", "md"])["cum_net_weight"]
          .transform(lambda s: s.expanding(min_periods=1).median().shift(1))
    )

    # Clean helper
    return jm.drop(columns=["md"])

def filter_inactive_rmids(full_daily: pd.DataFrame,
                          jan_may: pd.DataFrame,
                          lookback_years: int = LOOKBACK_YEARS,
                          prediction_mapping_path: str | None = None) -> pd.DataFrame:
    """
    Keep only (rm_id, year) rows where rm_id had at least one receival in the prior
    `lookback_years` before Jan 1 of that year. Optionally preserve any rm_id that
    appear in a prediction mapping file regardless of activity.
    """
    jm = jan_may.copy()

    # Build last-activity date per rm_id up to any cutoff
    # We will query this per-year at Jan 1 of Y (no peeking into the future).
    last_activity = (full_daily[full_daily["daily_net_weight"] > 0]
                     .groupby("rm_id", as_index=False)["date"].max()
                     .rename(columns={"date": "last_receival_date"}))

    # Optional: rm_id that must not be dropped (from prediction mapping)
    must_keep = set()
    if prediction_mapping_path is not None and Path(prediction_mapping_path).exists():
        pm = pd.read_csv(prediction_mapping_path, low_memory=False)
        if "rm_id" in pm.columns:
            must_keep = set(pm["rm_id"].dropna().astype("int64").unique())

    # For each year, compute the Jan 1 cutoff and decide active rm_id
    active_masks = []
    for y in sorted(jm["year"].unique().tolist()):
        jan1 = pd.Timestamp(year=y, month=1, day=1)
        cutoff = jan1 - pd.DateOffset(years=lookback_years)

        la = last_activity.copy()
        la["is_active"] = la["last_receival_date"] >= cutoff

        # rm_ids to keep for this year: active OR must_keep
        keep_ids = set(la.loc[la["is_active"], "rm_id"].astype("int64").tolist()) | must_keep

        mask_y = (jm["year"] == y) & (jm["rm_id"].isin(keep_ids))
        active_masks.append(mask_y)

    mask = np.logical_or.reduce(active_masks) if active_masks else np.array([], dtype=bool)
    jm = jm.loc[mask].copy()
    return jm

def sort_panel(jan_may: pd.DataFrame) -> pd.DataFrame:
    # Ensure deterministic time order within each rm_id-year
    return jan_may.sort_values(["rm_id", "year", "date"]).reset_index(drop=True)

def main():
    # Load and build features using your existing functions
    rec = load_clean("data_clean")
    full_daily = build_full_daily(rec)
    full_daily = add_calendar_features(full_daily)
    full_daily = add_rolling_cadence_features(full_daily)
    jan_may = build_supervised_panel(full_daily)
    jan_may = add_prior_year_features(jan_may)

    # Fill NA feature values (targets remain as computed)
    for col in ["prev_year_cum", "median_cum_same_day"]:
        if col in jan_may.columns:
            jan_may[col] = jan_may[col].fillna(0.0)

    # Sort by rm_id, year, date
    jan_may = sort_panel(jan_may)

    # Filter inactive rm_id with 5-year lookback per-year; preserve mapping rm_id if available
    # Set prediction_mapping_path to your mapping CSV if you want to force-keep those rm_id
    jan_may = filter_inactive_rmids(
        full_daily=full_daily,
        jan_may=jan_may,
        lookback_years=LOOKBACK_YEARS,
        prediction_mapping_path="prediction_mapping.csv"  # or None if not needed
    )

    # Save
    Path("data_model").mkdir(parents=True, exist_ok=True)
    jan_may.to_csv("data_model/panel_jan_may_features.csv", index=False)

if __name__ == "__main__":
    main()