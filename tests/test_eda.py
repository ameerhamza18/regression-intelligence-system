import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from regression_intelligence.analysis import eda


@pytest.fixture(autouse=True)
def _headless_matplotlib():
    matplotlib.use("Agg")
    yield
    plt.close("all")


def test_open_days_keeps_only_trading_rows(clean_df):
    out = eda.open_days(clean_df)
    assert (out["open"] == 1).all()
    assert (out["sales"] > 0).all()
    assert len(out) < len(clean_df)


def test_missing_summary_lists_only_columns_with_gaps(dirty_df):
    summary = eda.missing_summary(dirty_df)
    assert "competition_distance" in summary.index
    assert "sales" not in summary.index


def test_compute_vif_flags_collinear_columns():
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=500)
    df = pd.DataFrame(
        {"a": x1, "b": 2 * x1 + rng.normal(scale=0.01, size=500), "c": rng.normal(size=500)}
    )
    vif = eda.compute_vif(df, ["a", "b", "c"]).set_index("feature")["vif"]
    assert vif["a"] > 100
    assert vif["b"] > 100
    assert vif["c"] < 2


def test_log_transform_reduces_skew():
    sales = pd.Series(np.random.default_rng(0).lognormal(mean=8, sigma=0.6, size=2000))
    skew = eda.target_skew(sales)
    assert abs(skew["skew_log1p"]) < abs(skew["skew_raw"])


def test_plots_return_figures(clean_df):
    d = eda.open_days(clean_df)
    figures = [
        eda.plot_target_distribution(d),
        eda.plot_time_patterns(d),
        eda.plot_categorical_effects(d),
        eda.plot_competition_distance(d),
        eda.plot_store_heterogeneity(d),
        eda.plot_correlation(d, ["sales", "customers", "promo", "school_holiday"]),
    ]
    assert all(isinstance(f, Figure) for f in figures)


def test_save_fig_writes_png(tmp_path):
    path = eda.save_fig(plt.figure(), "demo", tmp_path)
    assert path.exists()
    assert path.suffix == ".png"
