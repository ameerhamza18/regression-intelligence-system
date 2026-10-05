import numpy as np
import pandas as pd
import pytest

from regression_intelligence.data.ingestion import load_rossmann, to_snake_case


def _train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Store": [1, 2, 1],
            "DayOfWeek": [5, 5, 4],
            "Date": ["2015-07-31", "2015-07-31", "2015-07-30"],
            "Sales": [5263, 6064, 5020],
            "Customers": [555, 625, 546],
            "Open": [1, 1, 1],
            "Promo": [1, 1, 1],
            "StateHoliday": ["0", "0", "a"],
            "SchoolHoliday": [1, 1, 1],
        }
    )


def _store_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Store": [1, 2],
            "StoreType": ["c", "a"],
            "Assortment": ["a", "a"],
            "CompetitionDistance": [1270, 570],
            "CompetitionOpenSinceMonth": [9, 11],
            "CompetitionOpenSinceYear": [2008, 2007],
            "Promo2": [0, 1],
            "Promo2SinceWeek": [np.nan, 13],
            "Promo2SinceYear": [np.nan, 2010],
            "PromoInterval": [np.nan, "Jan,Apr,Jul,Oct"],
        }
    )


@pytest.fixture
def raw_files(tmp_path):
    train, store = tmp_path / "train.csv", tmp_path / "store.csv"
    _train_frame().to_csv(train, index=False)
    _store_frame().to_csv(store, index=False)
    return train, store


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Store", "store"),
        ("DayOfWeek", "day_of_week"),
        ("Promo2SinceWeek", "promo2_since_week"),
        ("CompetitionOpenSinceMonth", "competition_open_since_month"),
        (" Store ID ", "store_id"),
    ],
)
def test_to_snake_case(raw, expected):
    assert to_snake_case(raw) == expected


def test_load_merges_and_normalises_columns(raw_files):
    df = load_rossmann(*raw_files)
    assert len(df) == 3
    assert {"day_of_week", "state_holiday", "competition_distance", "promo2_since_week"} <= set(
        df.columns
    )
    assert df.loc[df["store"] == 1, "store_type"].eq("c").all()


def test_load_parses_dates_and_sorts_chronologically(raw_files):
    df = load_rossmann(*raw_files)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].is_monotonic_increasing


def test_state_holiday_stays_text(raw_files):
    df = load_rossmann(*raw_files)
    assert set(df["state_holiday"]) == {"0", "a"}


def test_duplicate_store_rows_raise(tmp_path, raw_files):
    train, _ = raw_files
    store = _store_frame()
    bad_store = tmp_path / "bad_store.csv"
    pd.concat([store, store.iloc[[0]]]).to_csv(bad_store, index=False)
    with pytest.raises(pd.errors.MergeError):
        load_rossmann(train, bad_store)


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle"):
        load_rossmann(tmp_path / "train.csv", tmp_path / "store.csv")
