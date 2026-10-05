import json

import pytest

from regression_intelligence.data.validation import (
    DataValidationError,
    raise_if_critical,
    validate_dataframe,
)


def test_clean_data_passes(clean_df):
    report = validate_dataframe(clean_df)
    assert not report.has_critical
    assert report.warnings == []


def test_dirty_data_has_warnings_but_no_critical(dirty_df):
    report = validate_dataframe(dirty_df)
    checks = {i.check for i in report.warnings}
    assert not report.has_critical
    assert {
        "missing_values",
        "out_of_range",
        "invalid_category",
        "duplicate_rows",
        "closed_with_sales",
        "open_zero_sales",
    } <= checks


def test_closed_days_are_reported_as_info(clean_df):
    report = validate_dataframe(clean_df)
    assert any(i.check == "closed_days" for i in report.info)


def test_leakage_risk_is_flagged_for_customers(clean_df):
    report = validate_dataframe(clean_df)
    assert any(i.check == "leakage_risk" and i.column == "customers" for i in report.info)


def test_missing_column_is_critical(clean_df):
    report = validate_dataframe(clean_df.drop(columns=["sales"]))
    assert any(i.check == "missing_column" and i.column == "sales" for i in report.critical)


def test_wrong_dtype_is_critical(clean_df):
    df = clean_df.copy()
    df["sales"] = df["sales"].astype(str)
    report = validate_dataframe(df)
    assert any(i.check == "dtype" and i.column == "sales" for i in report.critical)


def test_too_few_rows_is_critical(clean_df):
    assert validate_dataframe(clean_df.head(10)).has_critical


def test_raise_if_critical(clean_df):
    report = validate_dataframe(clean_df.drop(columns=["sales"]))
    with pytest.raises(DataValidationError):
        raise_if_critical(report)


def test_report_is_json_serialisable(dirty_df):
    json.dumps(validate_dataframe(dirty_df).to_dict())
