"""Schema, data-quality and business-rule validation. Reports problems instead of crashing."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal

import pandas as pd
from pandas.api import types as ptypes

from regression_intelligence.utils.io import save_json
from regression_intelligence.utils.paths import METADATA_DIR

MIN_ROWS = 100
KEY_COLUMNS = ["date", "store"]
TARGET = "sales"
OUTLIER_FENCE = 3.0  # extreme-outlier fence: 3 x IQR beyond the quartiles


class DataValidationError(Exception):
    """Raised when the data is structurally unusable."""


class Severity(StrEnum):
    CRITICAL = "CRITICAL"  # structural problem: stop the pipeline
    WARNING = "WARNING"  # data-quality problem: handle in the cleaning stage
    INFO = "INFO"  # worth knowing, no action required


@dataclass(frozen=True)
class ColumnSpec:
    kind: Literal["numeric", "categorical", "datetime"]
    min: float | None = None
    max: float | None = None
    allowed: tuple[str, ...] | None = None


SCHEMA: dict[str, ColumnSpec] = {
    # train.csv
    "store": ColumnSpec("numeric", min=1, max=100_000),
    "day_of_week": ColumnSpec("numeric", min=1, max=7),
    "date": ColumnSpec("datetime"),
    "sales": ColumnSpec("numeric", min=0, max=1_000_000),
    "customers": ColumnSpec("numeric", min=0, max=100_000),
    "open": ColumnSpec("numeric", min=0, max=1),
    "promo": ColumnSpec("numeric", min=0, max=1),
    "state_holiday": ColumnSpec("categorical", allowed=("0", "a", "b", "c")),
    "school_holiday": ColumnSpec("numeric", min=0, max=1),
    # store.csv
    "store_type": ColumnSpec("categorical", allowed=("a", "b", "c", "d")),
    "assortment": ColumnSpec("categorical", allowed=("a", "b", "c")),
    "competition_distance": ColumnSpec("numeric", min=0, max=1_000_000),
    "competition_open_since_month": ColumnSpec("numeric", min=1, max=12),
    "competition_open_since_year": ColumnSpec("numeric", min=1900, max=2030),
    "promo2": ColumnSpec("numeric", min=0, max=1),
    "promo2_since_week": ColumnSpec("numeric", min=1, max=53),
    "promo2_since_year": ColumnSpec("numeric", min=2000, max=2030),
    "promo_interval": ColumnSpec("categorical"),
}


@dataclass(frozen=True)
class Issue:
    severity: Severity
    check: str
    message: str
    column: str | None = None
    count: int = 0


@dataclass
class ValidationReport:
    n_rows: int
    n_columns: int
    issues: list[Issue] = field(default_factory=list)

    @property
    def critical(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.CRITICAL]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def info(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.INFO]

    @property
    def has_critical(self) -> bool:
        return bool(self.critical)

    def summary(self) -> str:
        lines = [
            f"Rows: {self.n_rows:,} | Columns: {self.n_columns}",
            f"Critical: {len(self.critical)} | Warnings: {len(self.warnings)} "
            f"| Info: {len(self.info)}",
        ]
        for i in self.issues:
            where = f" ({i.column})" if i.column else ""
            lines.append(f"  [{i.severity}] {i.check}{where}: {i.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "passed": not self.has_critical,
            "issues": [asdict(i) for i in self.issues],
        }


def _kind_ok(s: pd.Series, kind: str) -> bool:
    if kind == "numeric":
        return ptypes.is_numeric_dtype(s)
    if kind == "datetime":
        return ptypes.is_datetime64_any_dtype(s)
    return (
        isinstance(s.dtype, pd.CategoricalDtype)
        or ptypes.is_object_dtype(s)
        or ptypes.is_string_dtype(s)
    )


def _check_column(df: pd.DataFrame, name: str, spec: ColumnSpec) -> list[Issue]:
    s = df[name]
    if not _kind_ok(s, spec.kind):
        return [Issue(Severity.CRITICAL, "dtype", f"expected {spec.kind}, got {s.dtype}", name)]

    issues: list[Issue] = []

    n_null = int(s.isna().sum())
    if n_null:
        share = n_null / len(df)
        issues.append(
            Issue(
                Severity.WARNING, "missing_values", f"{n_null} missing ({share:.1%})", name, n_null
            )
        )

    if spec.kind == "numeric":
        bad = pd.Series(False, index=s.index)
        if spec.min is not None:
            bad |= s < spec.min
        if spec.max is not None:
            bad |= s > spec.max
        n_bad = int(bad.sum())
        if n_bad:
            issues.append(
                Issue(
                    Severity.WARNING,
                    "out_of_range",
                    f"{n_bad} values outside [{spec.min}, {spec.max}]",
                    name,
                    n_bad,
                )
            )

    if spec.allowed is not None:
        n_bad = int((s.notna() & ~s.isin(spec.allowed)).sum())
        if n_bad:
            issues.append(
                Issue(
                    Severity.WARNING,
                    "invalid_category",
                    f"{n_bad} values not in {spec.allowed}",
                    name,
                    n_bad,
                )
            )
    return issues


def _check_dataset(df: pd.DataFrame, target: str) -> list[Issue]:
    issues: list[Issue] = []

    n_dup = int(df.duplicated().sum())
    if n_dup:
        issues.append(
            Issue(Severity.WARNING, "duplicate_rows", f"{n_dup} exact duplicate rows", count=n_dup)
        )

    if all(k in df.columns for k in KEY_COLUMNS):
        n_key = int(df.drop_duplicates().duplicated(subset=KEY_COLUMNS).sum())
        if n_key:
            issues.append(
                Issue(
                    Severity.WARNING,
                    "duplicate_keys",
                    f"{n_key} rows share {KEY_COLUMNS} but differ in values",
                    count=n_key,
                )
            )

    if target in df.columns and ptypes.is_numeric_dtype(df[target]):
        series = df[target]
        if "open" in df.columns and ptypes.is_numeric_dtype(df["open"]):
            series = series[df["open"] == 1]  # closed days are always 0, ignore them here
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - OUTLIER_FENCE * iqr, q3 + OUTLIER_FENCE * iqr
        n_out = int(((series < low) | (series > high)).sum())
        if n_out:
            issues.append(
                Issue(
                    Severity.INFO,
                    "target_outliers",
                    f"{n_out} extreme '{target}' values on open days (outside {OUTLIER_FENCE}xIQR)",
                    target,
                    n_out,
                )
            )
    return issues


def _check_business_rules(df: pd.DataFrame) -> list[Issue]:
    """Domain logic specific to retail daily sales."""
    issues: list[Issue] = []

    if (
        {"open", "sales"} <= set(df.columns)
        and ptypes.is_numeric_dtype(df["open"])
        and ptypes.is_numeric_dtype(df["sales"])
    ):
        closed = df["open"] == 0
        opened = df["open"] == 1

        n_closed = int(closed.sum())
        if n_closed:
            issues.append(
                Issue(
                    Severity.INFO,
                    "closed_days",
                    f"{n_closed} rows ({n_closed / len(df):.1%}) are closed days; "
                    "sales are 0 there, so drop them before modelling",
                    "open",
                    n_closed,
                )
            )

        n_closed_sales = int((closed & (df["sales"] > 0)).sum())
        if n_closed_sales:
            issues.append(
                Issue(
                    Severity.WARNING,
                    "closed_with_sales",
                    f"{n_closed_sales} closed-day rows have sales > 0",
                    "sales",
                    n_closed_sales,
                )
            )

        n_open_zero = int((opened & (df["sales"] == 0)).sum())
        if n_open_zero:
            issues.append(
                Issue(
                    Severity.WARNING,
                    "open_zero_sales",
                    f"{n_open_zero} open-day rows have zero sales",
                    "sales",
                    n_open_zero,
                )
            )

    if "customers" in df.columns:
        issues.append(
            Issue(
                Severity.INFO,
                "leakage_risk",
                "customers is only known after the day ends; exclude it from model features",
                "customers",
            )
        )

    if "date" in df.columns and ptypes.is_datetime64_any_dtype(df["date"]):
        d = df["date"].dropna()
        if not d.empty:
            issues.append(
                Issue(
                    Severity.INFO,
                    "date_range",
                    f"{d.min():%Y-%m-%d} to {d.max():%Y-%m-%d} ({d.nunique()} distinct days)",
                    "date",
                )
            )
    return issues


def validate_dataframe(
    df: pd.DataFrame,
    schema: dict[str, ColumnSpec] = SCHEMA,
    target: str = TARGET,
) -> ValidationReport:
    report = ValidationReport(n_rows=len(df), n_columns=df.shape[1])

    if len(df) < MIN_ROWS:
        report.issues.append(
            Issue(Severity.CRITICAL, "row_count", f"{len(df)} rows; need at least {MIN_ROWS}")
        )
        return report

    for col in schema:
        if col not in df.columns:
            report.issues.append(
                Issue(Severity.CRITICAL, "missing_column", "required column absent", col)
            )
    for col in df.columns:
        if col not in schema:
            report.issues.append(
                Issue(Severity.WARNING, "unexpected_column", "column not in schema", col)
            )

    for name, spec in schema.items():
        if name in df.columns:
            report.issues.extend(_check_column(df, name, spec))

    report.issues.extend(_check_dataset(df, target))
    report.issues.extend(_check_business_rules(df))
    return report


def raise_if_critical(report: ValidationReport) -> None:
    if report.has_critical:
        details = "; ".join(f"{i.check}:{i.column}" for i in report.critical)
        raise DataValidationError(f"Critical validation failures: {details}")


def save_report(report: ValidationReport, path: Path | None = None) -> Path:
    return save_json(report.to_dict(), path or METADATA_DIR / "validation_report.json")
