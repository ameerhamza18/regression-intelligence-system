"""Reusable EDA helpers: notebooks call these so analysis stays reproducible and testable."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from regression_intelligence.utils.paths import FIGURES_DIR

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def save_fig(fig: Figure, name: str, directory: Path = FIGURES_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


# ---------- tables ----------


def open_days(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where the store actually traded. Closed days are not demand and distort plots."""
    return df[(df["open"] == 1) & (df["sales"] > 0)].copy()


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    n = df.isna().sum()
    out = pd.DataFrame({"missing": n, "pct": (n / len(df) * 100).round(2)})
    return out[out["missing"] > 0].sort_values("pct", ascending=False)


def target_skew(sales: pd.Series) -> pd.Series:
    return pd.Series({"skew_raw": sales.skew(), "skew_log1p": np.log1p(sales).skew()})


def compute_vif(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Variance Inflation Factor: how well each feature is predicted by the others.

    Rule of thumb: > 5 worth a look, > 10 serious multicollinearity.
    """
    x = df[columns].dropna().to_numpy(dtype=float)
    vifs = []
    for i in range(len(columns)):
        y = x[:, i]
        design = np.column_stack([np.ones(len(x)), np.delete(x, i, axis=1)])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ coef
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
        vifs.append(1 / (1 - r2) if r2 < 1 else np.inf)
    return pd.DataFrame({"feature": columns, "vif": vifs}).sort_values("vif", ascending=False)


# ---------- plots ----------


def plot_target_distribution(df: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df["sales"], bins=60, color="#4C72B0")
    axes[0].set(title="Daily sales (open days)", xlabel="Sales", ylabel="Rows")
    axes[1].hist(np.log1p(df["sales"]), bins=60, color="#55A868")
    axes[1].set(title="log(1 + sales)", xlabel="log1p(Sales)", ylabel="Rows")
    fig.tight_layout()
    return fig


def plot_time_patterns(df: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    daily = df.groupby("date")["sales"].mean()
    axes[0, 0].plot(daily.index, daily.values, lw=0.6, alpha=0.6, label="daily")
    axes[0, 0].plot(daily.index, daily.rolling(28).mean(), lw=2, label="28-day mean")
    axes[0, 0].set(title="Average sales per open store-day", ylabel="Sales")
    axes[0, 0].legend()

    dow = df.groupby("day_of_week")["sales"].mean()
    axes[0, 1].bar([DAY_NAMES[d - 1] for d in dow.index], dow.values, color="#4C72B0")
    axes[0, 1].set(title="By day of week", ylabel="Mean sales")

    month = df.groupby(df["date"].dt.month)["sales"].mean()
    axes[1, 0].bar(month.index, month.values, color="#55A868")
    axes[1, 0].set_xticks(range(1, 13))
    axes[1, 0].set(title="By calendar month", xlabel="Month", ylabel="Mean sales")

    by_year = (
        df.assign(year=df["date"].dt.year, month=df["date"].dt.month)
        .groupby(["month", "year"])["sales"]
        .mean()
        .unstack("year")
    )
    by_year.plot(ax=axes[1, 1], marker="o")
    axes[1, 1].set(title="Monthly average by year", xlabel="Month", ylabel="Mean sales")

    fig.tight_layout()
    return fig


def plot_categorical_effects(df: pd.DataFrame, columns: list[str] | None = None) -> Figure:
    columns = columns or [
        "promo",
        "school_holiday",
        "state_holiday",
        "store_type",
        "assortment",
        "promo2",
    ]
    ncols = 3
    nrows = math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    flat = np.atleast_1d(axes).ravel()
    for ax, col in zip(flat, columns, strict=False):
        stats = df.groupby(col)["sales"].agg(["mean", "sem"])
        ax.bar(stats.index.astype(str), stats["mean"], yerr=1.96 * stats["sem"], color="#4C72B0")
        ax.set(title=f"Mean sales by {col}", ylabel="Mean sales")
    for ax in flat[len(columns) :]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_competition_distance(df: pd.DataFrame, n_bins: int = 10) -> Figure:
    d = df.dropna(subset=["competition_distance"])
    bins = pd.qcut(d["competition_distance"], q=n_bins, duplicates="drop")
    means = d.groupby(bins, observed=True)["sales"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(1, len(means) + 1), means.values, color="#4C72B0")
    axes[0].set(
        title="Mean sales by competition-distance bin (1 = closest)",
        xlabel="Bin",
        ylabel="Mean sales",
    )
    sample = d.sample(min(20_000, len(d)), random_state=42)
    axes[1].scatter(np.log1p(sample["competition_distance"]), sample["sales"], s=3, alpha=0.2)
    axes[1].set(title="Sales vs log(1 + competition distance)", xlabel="log1p(distance)")
    fig.tight_layout()
    return fig


def plot_store_heterogeneity(df: pd.DataFrame) -> Figure:
    per_store = df.groupby("store")["sales"].mean()
    days = df.groupby("store").size()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(per_store, bins=40, color="#4C72B0")
    axes[0].set(title=f"Mean sales per store ({len(per_store)} stores)", xlabel="Mean sales")
    axes[1].hist(days, bins=40, color="#55A868")
    axes[1].set(title="Open days per store", xlabel="Open days")
    fig.tight_layout()
    return fig


def plot_correlation(df: pd.DataFrame, columns: list[str]) -> Figure:
    corr = df[columns].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)), columns, rotation=45, ha="right")
    ax.set_yticks(range(len(columns)), columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.grid(False)
    ax.set_title("Correlation matrix (open days)")
    fig.tight_layout()
    return fig
