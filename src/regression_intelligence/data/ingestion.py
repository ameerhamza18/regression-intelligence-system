"""Load the Rossmann Store Sales tables, merge them, and record data lineage."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from regression_intelligence.utils.config import get_config
from regression_intelligence.utils.io import file_sha256, save_json
from regression_intelligence.utils.logging import get_logger
from regression_intelligence.utils.paths import METADATA_DIR, RAW_DIR

KAGGLE_URL = "https://www.kaggle.com/c/rossmann-store-sales/data"
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def to_snake_case(name: str) -> str:
    """'CompetitionOpenSinceMonth' -> 'competition_open_since_month'."""
    name = _CAMEL_BOUNDARY.sub("_", name.strip())
    return re.sub(r"\s+", "_", name).lower()


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}\n"
            f"Download train.csv and store.csv from {KAGGLE_URL} and place them in {path.parent}"
        )


def load_rossmann(train_path: Path, store_path: Path) -> pd.DataFrame:
    """Read both tables, standardise names, merge, and sort chronologically."""
    _require(train_path)
    _require(store_path)

    # StateHoliday mixes 0 and '0' in the raw file, so force it to text.
    train = pd.read_csv(train_path, dtype={"StateHoliday": str}, low_memory=False)
    store = pd.read_csv(store_path)
    train.columns = [to_snake_case(c) for c in train.columns]
    store.columns = [to_snake_case(c) for c in store.columns]

    if "date" in train.columns:
        # Unparseable dates become NaT; validation reports them instead of crashing here.
        train["date"] = pd.to_datetime(train["date"], errors="coerce")

    # many_to_one: a duplicated store row would silently duplicate sales rows.
    df = train.merge(store, on="store", how="left", validate="many_to_one")
    return df.sort_values(["date", "store"]).reset_index(drop=True)


def ingest(train_filename: str | None = None, store_filename: str | None = None) -> pd.DataFrame:
    cfg = get_config()
    logger = get_logger(__name__, cfg.logging.level)

    train_path = RAW_DIR / (train_filename or cfg.data.train_filename)
    store_path = RAW_DIR / (store_filename or cfg.data.store_filename)

    df = load_rossmann(train_path, store_path)
    logger.info("Loaded %d rows x %d columns", len(df), df.shape[1])

    manifest = {
        "source": KAGGLE_URL,
        "files": {p.name: file_sha256(p) for p in (train_path, store_path)},
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "columns": list(df.columns),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "ingested_at": datetime.now(UTC).isoformat(),
    }
    save_json(manifest, METADATA_DIR / "ingestion_manifest.json")
    return df
