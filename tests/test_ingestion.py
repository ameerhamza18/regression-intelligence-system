import pandas as pd
import pytest

from regression_intelligence.data.ingestion import load_rossmann, to_snake_case

TRAIN_CSV = """Store,DayOfWeek,Date,Sales,Customers,Open,Promo,StateHoliday,SchoolHoliday
1,5,2015-07-31,5263,555,1,1,0,1
2,5,2015-07-31,6064,625,1,1,0,1
1,4,2015-07-30,5020,546,1,1,a,1
"""

STORE_CSV = """Store,StoreType,Assortment,CompetitionDistance,
CompetitionOpenSinceMonth,CompetitionOpenSinceYear,
Promo2,Promo2SinceWeek,Promo2SinceYear,PromoInterval
1,c,a,1270,9,2008,0,,,
2,a,a,570,11,2007,1,13,2010,"Jan,Apr,Jul,Oct"
"""


@pytest.fixture
def raw_files(tmp_path):
    train, store = tmp_path / "train.csv", tmp_path / "store.csv"
    train.write_text(TRAIN_CSV)
    store.write_text(STORE_CSV)
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
    bad_store = tmp_path / "bad_store.csv"
    bad_store.write_text(STORE_CSV + "1,c,a,1270,9,2008,0,,,\n")
    with pytest.raises(pd.errors.MergeError):
        load_rossmann(train, bad_store)


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="kaggle"):
        load_rossmann(tmp_path / "train.csv", tmp_path / "store.csv")
