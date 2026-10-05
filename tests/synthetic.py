"""Tiny Rossmann-shaped fake data so tests never need the real Kaggle files."""

import numpy as np
import pandas as pd


def make_rossmann_like(
    n_stores: int = 5, n_days: int = 60, seed: int = 0, inject_issues: bool = False
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stores = pd.DataFrame(
        {
            "store": np.arange(1, n_stores + 1),
            "store_type": rng.choice(list("abcd"), n_stores),
            "assortment": rng.choice(list("abc"), n_stores),
            "competition_distance": rng.uniform(100, 20_000, n_stores).round(),
            "competition_open_since_month": rng.integers(1, 13, n_stores),
            "competition_open_since_year": rng.integers(2000, 2013, n_stores),
            "promo2": 1,
            "promo2_since_week": rng.integers(1, 53, n_stores),
            "promo2_since_year": rng.integers(2009, 2014, n_stores),
            "promo_interval": "Jan,Apr,Jul,Oct",
        }
    )
    dates = pd.date_range("2014-01-01", periods=n_days, freq="D")
    df = stores.merge(pd.DataFrame({"date": dates}), how="cross")
    n = len(df)

    df["day_of_week"] = df["date"].dt.dayofweek + 1
    df["open"] = (df["day_of_week"] != 7).astype(int)  # closed on Sundays
    df["promo"] = (rng.random(n) < 0.4).astype(int)
    df["state_holiday"] = "0"
    df["school_holiday"] = (rng.random(n) < 0.2).astype(int)

    base = 5000 + 800 * df["promo"] + 300 * df["school_holiday"] + rng.normal(0, 400, n)
    is_open = df["open"] == 1
    df["sales"] = np.where(is_open, base.clip(lower=100), 0).round().astype(int)
    df["customers"] = np.where(is_open, base / 9, 0).round().astype(int)

    df = df.sort_values(["date", "store"]).reset_index(drop=True)
    return _inject_issues(df, rng) if inject_issues else df


def _inject_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    open_idx = rng.permutation(df.index[df["open"] == 1].to_numpy())
    closed_idx = rng.permutation(df.index[df["open"] == 0].to_numpy())
    any_idx = rng.permutation(n)

    df.loc[open_idx[:5], "sales"] = -50  # negative sales
    df.loc[open_idx[5:9], "sales"] = 0  # open but zero sales
    df.loc[closed_idx[:5], "sales"] = 3000  # closed but has sales
    df.loc[any_idx[:10], "competition_distance"] = np.nan
    df.loc[any_idx[10:13], "store_type"] = "z"  # invalid category
    return pd.concat([df, df.iloc[any_idx[13:19]]], ignore_index=True)  # 6 duplicate rows
