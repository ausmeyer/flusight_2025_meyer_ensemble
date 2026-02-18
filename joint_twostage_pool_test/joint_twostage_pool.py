#!/usr/bin/env python
"""
Joint two-stage pooled-horizon bagged forecasting (isolated test module).

Implements:
- Joint fitting across locations
- Pooled horizons in one model table
- Season-level bagging
- Stage 1: LightGBM on log-target (mu)
- Stage 2: LightGBMLSS frozen-mu bounded sigma

Modes:
- backtest: expanding-anchor forecasts from start date to end of data
- prospective: latest-anchor forecast
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

try:
    from lightgbmlss.model import LightGBMLSS
except ImportError as exc:
    raise ImportError("lightgbmlss is required for this script.") from exc


# Ensure local imports from this repository's src/ utilities.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils.distributions import (  # noqa: E402
    GaussianFrozenLoc,
    GaussianFrozenLocBounded,
    GaussianFrozenLocBoundedWide,
)


TARGET_NAME = "wk inc flu hosp"
QUANTILES = np.array(
    [
        0.01,
        0.025,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        0.975,
        0.99,
    ],
    dtype=float,
)


@dataclass
class RuntimeConfig:
    data_file: str
    output: str
    max_horizons: int
    num_bags: int
    bag_frac: float
    seed: int
    sigma_mode: str
    target_mode: str
    stage1_rounds: int
    stage2_rounds: int
    own_lags: List[int]
    cov_top_k: int
    cov_lags: List[int]
    cov_min_overlap: int
    locations: Optional[List[str]]
    min_train_rows: int


def infer_flu_season(ts: pd.Timestamp) -> str:
    start_year = ts.year if ts.month >= 10 else ts.year - 1
    return f"{start_year}/{str(start_year + 1)[-2:]}"


def parse_lag_string(lag_str: str) -> List[int]:
    out = []
    for token in lag_str.split(","):
        token = token.strip()
        if not token:
            continue
        lag = int(token)
        if lag < 1:
            raise ValueError(f"Invalid lag {lag}; all lags must be >= 1.")
        out.append(lag)
    if not out:
        raise ValueError("At least one lag must be provided.")
    return sorted(set(out))


def load_location_fips_map(path: str = "data/locations.csv") -> Dict[str, str]:
    loc = pd.read_csv(path, dtype={"location": str})
    if "location_name" not in loc.columns or "location" not in loc.columns:
        raise ValueError(f"Expected location_name/location in {path}")
    return dict(zip(loc["location_name"], loc["location"]))


def load_stitched_long(data_file: str, locations: Optional[Sequence[str]] = None) -> pd.DataFrame:
    df = pd.read_csv(data_file)
    expected = {"location_name", "date", "total_hosp"}
    if not expected.issubset(df.columns):
        raise ValueError(f"{data_file} missing required columns: {sorted(expected)}")

    df = df.loc[:, ["location_name", "date", "total_hosp"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["total_hosp"] = pd.to_numeric(df["total_hosp"], errors="coerce")
    df = df.dropna(subset=["location_name", "date", "total_hosp"])

    if locations:
        keep = set(locations)
        df = df[df["location_name"].isin(keep)].copy()
        if df.empty:
            raise ValueError("No rows remain after applying --locations filter.")

    # If duplicates exist at location/date, average them.
    df = (
        df.groupby(["location_name", "date"], as_index=False)["total_hosp"]
        .mean()
        .sort_values(["location_name", "date"])
        .reset_index(drop=True)
    )
    return df


def compute_top_cov_donors(
    df_long: pd.DataFrame,
    top_k: int,
    min_overlap: int,
    target_mode: str,
) -> Dict[str, List[str]]:
    if top_k <= 0:
        return {}

    pivot = df_long.pivot(index="date", columns="location_name", values="total_hosp").sort_index()
    # Normalize by scale before donor selection:
    # - always work in log space to reduce size effects
    # - in delta_log mode, select donors on week-to-week log changes
    pivot_log = np.log1p(np.clip(pivot.astype(float), 0.0, None))
    if target_mode == "delta_log":
        pivot_sel = pivot_log.diff(1)
    else:
        pivot_sel = pivot_log

    locations = list(pivot.columns)
    donor_map: Dict[str, List[str]] = {}

    for loc in locations:
        if loc == "US":
            candidates = [c for c in locations if c != "US"]
        else:
            # Skip US for non-US targets (already represented by global features).
            candidates = [c for c in locations if c not in (loc, "US")]
            if not candidates:
                candidates = [c for c in locations if c != loc]

        cov_scores: List[Tuple[str, float]] = []
        target = pivot_sel[loc]
        for cand in candidates:
            pair = pd.concat([target, pivot_sel[cand]], axis=1).dropna()
            if len(pair) < min_overlap:
                continue
            std_a = float(pair.iloc[:, 0].std())
            std_b = float(pair.iloc[:, 1].std())
            if std_a <= 1e-12 or std_b <= 1e-12:
                continue
            corr_val = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
            if np.isnan(corr_val):
                continue
            cov_scores.append((cand, corr_val))

        cov_scores.sort(key=lambda x: x[1], reverse=True)
        donor_map[loc] = [name for name, _ in cov_scores[:top_k]]

    return donor_map


def add_donor_covariate_features(
    df_long: pd.DataFrame,
    donor_map: Dict[str, List[str]],
    donor_lags: Sequence[int],
    top_k: int,
) -> pd.DataFrame:
    if top_k <= 0 or not donor_lags:
        return df_long

    pivot = df_long.pivot(index="date", columns="location_name", values="total_hosp").sort_index()
    parts = []
    for loc, grp in df_long.groupby("location_name", sort=False):
        g = grp.copy()
        donors = donor_map.get(loc, [])
        for rank in range(1, top_k + 1):
            donor = donors[rank - 1] if rank - 1 < len(donors) else None
            for lag in donor_lags:
                feat = f"cov_state{rank}_lag_{lag}"
                if donor is None or donor not in pivot.columns:
                    g[feat] = np.nan
                else:
                    g[feat] = pivot[donor].shift(lag).reindex(g["date"]).to_numpy()
        parts.append(g)

    out = pd.concat(parts, axis=0, ignore_index=True)
    out = out.sort_values(["location_name", "date"]).reset_index(drop=True)
    return out


def add_global_features(df_long: pd.DataFrame, lag_set: Sequence[int]) -> pd.DataFrame:
    pivot = df_long.pivot(index="date", columns="location_name", values="total_hosp").sort_index()
    if "US" in pivot.columns:
        us = pivot["US"]
    else:
        us = pivot.mean(axis=1)

    states_only = pivot.drop(columns=["US"], errors="ignore")
    if states_only.shape[1] == 0:
        nat_mean = us.copy()
        nat_std = pd.Series(0.0, index=pivot.index)
    else:
        nat_mean = states_only.mean(axis=1)
        nat_std = states_only.std(axis=1)

    global_df = pd.DataFrame(index=pivot.index)
    for lag in lag_set:
        global_df[f"us_lag_{lag}"] = us.shift(lag)
        global_df[f"nat_mean_lag_{lag}"] = nat_mean.shift(lag)
    global_df["nat_std_lag_1"] = nat_std.shift(1)
    global_df["nat_mean_chg_1"] = nat_mean.diff(1).shift(1)
    global_df["nat_mean_chg_4"] = nat_mean.diff(4).shift(1)
    global_df = global_df.reset_index()

    return df_long.merge(global_df, on="date", how="left")


def build_feature_table(
    df_long: pd.DataFrame,
    own_lags: Sequence[int],
    donor_map: Dict[str, List[str]],
    donor_lags: Sequence[int],
    donor_top_k: int,
) -> pd.DataFrame:
    df = df_long.copy().sort_values(["location_name", "date"]).reset_index(drop=True)
    g = df.groupby("location_name", group_keys=False)

    # Own lag features.
    for lag in own_lags:
        df[f"y_lag_{lag}"] = g["total_hosp"].shift(lag)

    # Simple local dynamics.
    df["y_diff_1"] = g["total_hosp"].diff(1)
    df["y_diff_4"] = g["total_hosp"].diff(4)
    df["y_pct_chg_1"] = g["total_hosp"].pct_change(1).replace([np.inf, -np.inf], np.nan)
    df["y_pct_chg_4"] = g["total_hosp"].pct_change(4).replace([np.inf, -np.inf], np.nan)

    # Rolling stats using only information up to t-1.
    for window in (2, 4, 8, 12):
        df[f"y_roll_mean_{window}"] = g["total_hosp"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"y_roll_std_{window}"] = g["total_hosp"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=2).std()
        )

    # Date/seasonality features.
    day_of_year = df["date"].dt.dayofyear.astype(float)
    year_phase = day_of_year / 365.25
    df["week_sin"] = np.sin(2 * np.pi * year_phase)
    df["week_cos"] = np.cos(2 * np.pi * year_phase)
    df["quarter_sin"] = np.sin(2 * np.pi * year_phase * 4.0)
    df["quarter_cos"] = np.cos(2 * np.pi * year_phase * 4.0)
    month = df["date"].dt.month
    df["is_flu_season"] = ((month >= 10) | (month <= 3)).astype(int)
    df["is_peak_flu"] = month.isin([12, 1, 2]).astype(int)
    # Three data-regime covariates:
    # regime1 = pre-2022/23, regime2 = 2022/23-2023/24, regime3 = post-2024 restart.
    regime2_start = pd.Timestamp("2022-07-01")
    regime3_start = pd.Timestamp("2024-11-01")
    df["regime_pre_2022_23"] = (df["date"] < regime2_start).astype(int)
    df["regime_2022_23_to_2023_24"] = ((df["date"] >= regime2_start) & (df["date"] < regime3_start)).astype(int)
    df["regime_post_2024_11"] = (df["date"] >= regime3_start).astype(int)
    weeks_from_regime2 = (df["date"] - regime2_start).dt.days.astype(float) / 7.0
    weeks_from_regime3 = (df["date"] - regime3_start).dt.days.astype(float) / 7.0
    df["weeks_since_regime2_start"] = np.log1p(np.clip(weeks_from_regime2, 0.0, None))
    df["weeks_since_regime3_start"] = np.log1p(np.clip(weeks_from_regime3, 0.0, None))
    df["weeks_before_regime3_start"] = np.log1p(np.clip(-weeks_from_regime3, 0.0, None))
    df["regime2_x_flu_season"] = df["regime_2022_23_to_2023_24"] * df["is_flu_season"]
    df["regime3_x_flu_season"] = df["regime_post_2024_11"] * df["is_flu_season"]
    df["regime2_x_week_sin"] = df["regime_2022_23_to_2023_24"] * df["week_sin"]
    df["regime2_x_week_cos"] = df["regime_2022_23_to_2023_24"] * df["week_cos"]
    df["regime3_x_week_sin"] = df["regime_post_2024_11"] * df["week_sin"]
    df["regime3_x_week_cos"] = df["regime_post_2024_11"] * df["week_cos"]

    # Global features.
    df = add_global_features(df, lag_set=sorted(set([1, 2, 4, 8, 12, 52] + list(own_lags))))
    df = add_donor_covariate_features(df, donor_map=donor_map, donor_lags=donor_lags, top_k=donor_top_k)

    # Location one-hot (joint model with location identity).
    loc_dummies = pd.get_dummies(df["location_name"], prefix="loc", dtype=float)
    df = pd.concat([df, loc_dummies], axis=1)

    # Flu season label for season-level bagging.
    df["season"] = df["date"].map(infer_flu_season)
    return df


def build_pooled_examples(feature_df: pd.DataFrame, max_horizons: int) -> pd.DataFrame:
    parts = []
    for h in range(1, max_horizons + 1):
        tmp = feature_df.copy()
        tmp["horizon_weeks"] = h
        tmp["target"] = tmp.groupby("location_name")["total_hosp"].shift(-h)
        tmp["target_date"] = tmp["date"] + pd.Timedelta(weeks=h)
        tmp["horizon_sin"] = np.sin(2 * np.pi * h / 8.0)
        tmp["horizon_cos"] = np.cos(2 * np.pi * h / 8.0)
        parts.append(tmp)
    out = pd.concat(parts, axis=0, ignore_index=True)
    return out


def get_feature_columns(pooled_df: pd.DataFrame) -> List[str]:
    excluded = {"location_name", "date", "total_hosp", "target", "target_date", "season"}
    cols = [c for c in pooled_df.columns if c not in excluded]
    return cols


def distribution_for_mode(sigma_mode: str):
    if sigma_mode == "wide":
        return GaussianFrozenLocBoundedWide(sigma_min=0.1, sigma_max=0.55)
    if sigma_mode == "narrow":
        return GaussianFrozenLocBounded()
    if sigma_mode == "unbounded":
        return GaussianFrozenLoc()
    raise ValueError(f"Unknown sigma_mode={sigma_mode}")


def fit_two_stage_one_bag(
    X_train: pd.DataFrame,
    y_train_model: np.ndarray,
    stage1_rounds: int,
    stage2_rounds: int,
    sigma_mode: str,
    seed: int,
) -> Tuple[lgb.Booster, LightGBMLSS]:
    X = X_train.astype(float)
    y = np.asarray(y_train_model, dtype=float)

    p1 = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
        "random_state": int(seed),
    }
    d1 = lgb.Dataset(X, label=y, params={"verbose": -1})
    stage1 = lgb.train(p1, d1, num_boost_round=stage1_rounds, callbacks=[])

    mu_train = stage1.predict(X)
    init_score = np.column_stack([mu_train, np.zeros_like(mu_train)]).ravel(order="F")

    p2 = {
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 0.1,
        "feature_pre_filter": False,
        "force_col_wise": True,
        "verbosity": -1,
        "random_state": int(seed),
    }
    d2 = lgb.Dataset(X, label=y, init_score=init_score, params={"verbose": -1}, free_raw_data=False)
    stage2 = LightGBMLSS(distribution_for_mode(sigma_mode))
    stage2.start_values = np.array([float(np.mean(mu_train)), 0.0], dtype=np.float32)
    stage2.train(p2, d2, num_boost_round=stage2_rounds)

    return stage1, stage2


def _extract_sigma(dist_params: np.ndarray, n_rows: int) -> np.ndarray:
    params = dist_params.values if hasattr(dist_params, "values") else np.asarray(dist_params)
    if params.ndim == 1:
        if params.shape[0] == n_rows:
            sigma = params
        elif params.shape[0] == 2:
            sigma = np.repeat(params[-1], n_rows)
        else:
            sigma = np.repeat(float(params[-1]), n_rows)
    else:
        if params.shape[1] == 1:
            sigma = params[:, 0]
        else:
            sigma = params[:, -1]
    return np.maximum(np.asarray(sigma, dtype=float), 1e-6)


def predict_quantiles(
    stage1: lgb.Booster,
    stage2: LightGBMLSS,
    test_X: pd.DataFrame,
    quantiles: np.ndarray,
    target_mode: str,
    current_obs: np.ndarray,
) -> np.ndarray:
    X = test_X.astype(float)
    mu_model = stage1.predict(X)
    dist_params = stage2.predict(X, pred_type="parameters")
    sigma_model = _extract_sigma(dist_params, len(X))

    q_model = norm.ppf(quantiles[None, :], loc=mu_model[:, None], scale=sigma_model[:, None])
    if target_mode == "level":
        q_log = q_model
    elif target_mode == "delta_log":
        base_log = np.log1p(np.clip(np.asarray(current_obs, dtype=float), 0.0, None))
        q_log = q_model + base_log[:, None]
    else:
        raise ValueError(f"Unknown target_mode={target_mode}")

    q_lin = np.expm1(q_log)
    return np.maximum(q_lin, 0.0)


def forecast_for_anchor(
    pooled_df: pd.DataFrame,
    feature_cols: Sequence[str],
    anchor_date: pd.Timestamp,
    cfg: RuntimeConfig,
    fips_map: Dict[str, str],
) -> pd.DataFrame:
    train_df = pooled_df[(pooled_df["target_date"] <= anchor_date) & (pooled_df["date"] <= anchor_date)].copy()
    test_df = pooled_df[(pooled_df["date"] == anchor_date) & (pooled_df["horizon_weeks"] <= cfg.max_horizons)].copy()

    if train_df.empty or test_df.empty:
        return pd.DataFrame()

    # Remove rows with missing target/features.
    train_df = train_df.dropna(subset=["target"] + list(feature_cols))
    test_df = test_df.dropna(subset=list(feature_cols))
    if len(train_df) < cfg.min_train_rows or test_df.empty:
        return pd.DataFrame()

    train_target = np.log1p(np.clip(train_df["target"].to_numpy(dtype=float), 0.0, None))
    if cfg.target_mode == "delta_log":
        base_log = np.log1p(np.clip(train_df["total_hosp"].to_numpy(dtype=float), 0.0, None))
        train_target = train_target - base_log
    elif cfg.target_mode != "level":
        raise ValueError(f"Unknown target_mode={cfg.target_mode}")
    valid_target = np.isfinite(train_target)
    train_df = train_df.loc[valid_target].reset_index(drop=True)
    train_target = train_target[valid_target]
    if len(train_df) < cfg.min_train_rows:
        return pd.DataFrame()

    seasons = sorted(train_df["season"].dropna().unique())
    if not seasons:
        return pd.DataFrame()

    bag_size = max(1, int(round(len(seasons) * cfg.bag_frac)))
    rng_seed = int(cfg.seed + int(anchor_date.value // 10**9))
    rng = np.random.default_rng(rng_seed)

    bag_preds = []
    good_bags = 0

    for b in range(cfg.num_bags):
        sampled = rng.choice(seasons, size=bag_size, replace=False)
        bag_mask = train_df["season"].isin(sampled).to_numpy()
        bag_train = train_df.loc[bag_mask]
        bag_target = train_target[bag_mask]
        if len(bag_train) < cfg.min_train_rows:
            continue
        try:
            stage1, stage2 = fit_two_stage_one_bag(
                X_train=bag_train.loc[:, feature_cols],
                y_train_model=bag_target,
                stage1_rounds=cfg.stage1_rounds,
                stage2_rounds=cfg.stage2_rounds,
                sigma_mode=cfg.sigma_mode,
                seed=rng_seed + b,
            )
            q = predict_quantiles(
                stage1,
                stage2,
                test_df.loc[:, feature_cols],
                QUANTILES,
                target_mode=cfg.target_mode,
                current_obs=test_df["total_hosp"].to_numpy(dtype=float),
            )
            bag_preds.append(q)
            good_bags += 1
        except Exception:
            continue

    if good_bags == 0:
        return pd.DataFrame()

    bag_tensor = np.stack(bag_preds, axis=0)  # [bags, rows, quantiles]
    agg_q = np.median(bag_tensor, axis=0)

    ref_date = anchor_date + pd.Timedelta(weeks=1)
    records = []
    for i, row in test_df.reset_index(drop=True).iterrows():
        loc_name = row["location_name"]
        loc_fips = fips_map.get(loc_name, loc_name)
        h = int(row["horizon_weeks"])
        for q_idx, q_level in enumerate(QUANTILES):
            records.append(
                {
                    "reference_date": ref_date.date().isoformat(),
                    "target": TARGET_NAME,
                    "horizon": h - 1,
                    "target_end_date": row["target_date"].date().isoformat(),
                    "location": loc_fips,
                    "output_type": "quantile",
                    "output_type_id": float(q_level),
                    "value": float(max(0.0, agg_q[i, q_idx])),
                }
            )

    return pd.DataFrame.from_records(records)


def run_backtest(
    cfg: RuntimeConfig,
    start_date: str,
    anchor_step_weeks: int,
    max_anchors: Optional[int],
    include_partial_horizons: bool = False,
) -> pd.DataFrame:
    df_long = load_stitched_long(cfg.data_file, locations=cfg.locations)
    all_dates = sorted(df_long["date"].unique())
    start_ts = pd.to_datetime(start_date)
    donor_source = df_long[df_long["date"] < start_ts]
    donor_fit_df = donor_source if len(donor_source) > 0 else df_long
    donor_map = compute_top_cov_donors(
        donor_fit_df,
        top_k=cfg.cov_top_k,
        min_overlap=cfg.cov_min_overlap,
        target_mode=cfg.target_mode,
    )
    print(
        f"Donor-state features (normalized): top_k={cfg.cov_top_k}, "
        f"lags={cfg.cov_lags}, overlap>={cfg.cov_min_overlap} weeks, target_mode={cfg.target_mode}"
    )
    feature_df = build_feature_table(
        df_long,
        own_lags=cfg.own_lags,
        donor_map=donor_map,
        donor_lags=cfg.cov_lags,
        donor_top_k=cfg.cov_top_k,
    )
    pooled_df = build_pooled_examples(feature_df, cfg.max_horizons)
    feature_cols = get_feature_columns(pooled_df)
    fips_map = load_location_fips_map()

    if include_partial_horizons:
        last_anchor = all_dates[-1]
    else:
        last_anchor = all_dates[-1] - pd.Timedelta(weeks=cfg.max_horizons)

    anchors = [d for d in all_dates if d >= start_ts and d <= last_anchor]
    if anchor_step_weeks > 1:
        anchors = anchors[::anchor_step_weeks]
    if max_anchors is not None and max_anchors > 0:
        anchors = anchors[: max_anchors]

    out_parts = []
    total = len(anchors)
    print(f"Backtest anchors: {total}")
    for i, anchor in enumerate(anchors, start=1):
        print(f"[{i}/{total}] anchor={anchor.date().isoformat()}")
        pred = forecast_for_anchor(
            pooled_df=pooled_df,
            feature_cols=feature_cols,
            anchor_date=anchor,
            cfg=cfg,
            fips_map=fips_map,
        )
        if not pred.empty:
            out_parts.append(pred)

    if not out_parts:
        return pd.DataFrame(
            columns=[
                "reference_date",
                "target",
                "horizon",
                "target_end_date",
                "location",
                "output_type",
                "output_type_id",
                "value",
            ]
        )
    return pd.concat(out_parts, axis=0, ignore_index=True)


def run_prospective(cfg: RuntimeConfig) -> pd.DataFrame:
    df_long = load_stitched_long(cfg.data_file, locations=cfg.locations)
    donor_map = compute_top_cov_donors(
        df_long,
        top_k=cfg.cov_top_k,
        min_overlap=cfg.cov_min_overlap,
        target_mode=cfg.target_mode,
    )
    print(
        f"Donor-state features (normalized): top_k={cfg.cov_top_k}, "
        f"lags={cfg.cov_lags}, overlap>={cfg.cov_min_overlap} weeks, target_mode={cfg.target_mode}"
    )
    feature_df = build_feature_table(
        df_long,
        own_lags=cfg.own_lags,
        donor_map=donor_map,
        donor_lags=cfg.cov_lags,
        donor_top_k=cfg.cov_top_k,
    )
    pooled_df = build_pooled_examples(feature_df, cfg.max_horizons)
    feature_cols = get_feature_columns(pooled_df)
    fips_map = load_location_fips_map()

    anchor = pd.to_datetime(df_long["date"].max())
    print(f"Prospective anchor={anchor.date().isoformat()}")
    return forecast_for_anchor(
        pooled_df=pooled_df,
        feature_cols=feature_cols,
        anchor_date=anchor,
        cfg=cfg,
        fips_map=fips_map,
    )


def write_output(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved: {path} ({len(df)} rows)")


def build_common_config(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig(
        data_file=args.data_file,
        output=args.output,
        max_horizons=args.max_horizons,
        num_bags=args.num_bags,
        bag_frac=args.bag_frac,
        seed=args.seed,
        sigma_mode=args.sigma_mode,
        target_mode=args.target_mode,
        stage1_rounds=args.stage1_rounds,
        stage2_rounds=args.stage2_rounds,
        own_lags=parse_lag_string(args.own_lags),
        cov_top_k=args.cov_top_k,
        cov_lags=parse_lag_string(args.cov_lags),
        cov_min_overlap=args.cov_min_overlap,
        locations=args.locations,
        min_train_rows=args.min_train_rows,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Joint two-stage pooled-horizon bagged forecasting test module")
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data-file", required=True, help="Path to stitched dataset CSV")
        p.add_argument("--output", required=True, help="Output forecast CSV")
        p.add_argument("--max-horizons", type=int, default=4, help="Number of pooled horizons (default: 4)")
        p.add_argument("--num-bags", type=int, default=100, help="Number of season bags (default: 100)")
        p.add_argument("--bag-frac", type=float, default=0.7, help="Fraction of seasons per bag (default: 0.7)")
        p.add_argument("--seed", type=int, default=2026, help="Random seed")
        p.add_argument(
            "--sigma-mode",
            choices=["narrow", "wide", "unbounded"],
            default="wide",
            help="Stage-2 sigma distribution mode",
        )
        p.add_argument(
            "--target-mode",
            choices=["level", "delta_log"],
            default="level",
            help="Training target parameterization: level (log target) or delta_log (log change from current level)",
        )
        p.add_argument("--stage1-rounds", type=int, default=200, help="Boosting rounds for stage 1")
        p.add_argument("--stage2-rounds", type=int, default=150, help="Boosting rounds for stage 2")
        p.add_argument(
            "--own-lags",
            type=str,
            default="1,2,3,4,5,6,7,8,9,10,11,12,52",
            help="Comma-separated own-lag list",
        )
        p.add_argument(
            "--cov-top-k",
            type=int,
            default=5,
            help="Number of donor states per location selected by covariance (default: 5)",
        )
        p.add_argument(
            "--cov-lags",
            type=str,
            default="1,2,3,4,8,12,52",
            help="Comma-separated lag list for donor-state covariates",
        )
        p.add_argument(
            "--cov-min-overlap",
            type=int,
            default=40,
            help="Minimum overlapping weeks required to compute covariance (default: 40)",
        )
        p.add_argument(
            "--locations",
            nargs="+",
            default=None,
            help="Optional subset of location names (e.g., California Texas US)",
        )
        p.add_argument(
            "--min-train-rows",
            type=int,
            default=1000,
            help="Minimum rows required for a bag to be fit",
        )

    p_backtest = sub.add_parser("backtest", help="Run expanding-anchor backtest")
    add_common(p_backtest)
    p_backtest.add_argument("--start-date", required=True, help="First anchor date to backtest (YYYY-MM-DD)")
    p_backtest.add_argument("--anchor-step-weeks", type=int, default=1, help="Anchor stride in weeks")
    p_backtest.add_argument("--max-anchors", type=int, default=None, help="Optional cap on number of anchors")
    p_backtest.add_argument(
        "--include-partial-horizons",
        action="store_true",
        help="Run anchors through the final observed date, even when later horizons lack realized truth yet.",
    )

    p_pros = sub.add_parser("prospective", help="Run latest-anchor prospective forecast")
    add_common(p_pros)

    return parser


def main() -> None:
    args = make_parser().parse_args()
    cfg = build_common_config(args)

    if args.mode == "backtest":
        result = run_backtest(
            cfg=cfg,
            start_date=args.start_date,
            anchor_step_weeks=args.anchor_step_weeks,
            max_anchors=args.max_anchors,
            include_partial_horizons=args.include_partial_horizons,
        )
        write_output(result, cfg.output)
        return

    if args.mode == "prospective":
        result = run_prospective(cfg=cfg)
        write_output(result, cfg.output)
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
