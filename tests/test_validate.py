import pytest

from eta.data import TARGET
from eta.features import CATEGORICAL, NUMERIC, build_table
from eta.validate import DataValidationError, validate_raw, validate_table


def test_validate_raw_passes_clean_data(raw):
    summary = validate_raw(raw)
    assert summary["rows"] == len(raw)
    assert summary["date_min"] == "2022-03-01"


def test_validate_raw_reports_every_problem(raw):
    bad = raw.copy()
    bad.loc[0, "courier_rating"] = 9.0
    bad.loc[1, "order_id"] = bad.loc[2, "order_id"]
    bad.loc[3, "traffic"] = "Gridlock"
    with pytest.raises(DataValidationError) as err:
        validate_raw(bad)
    msg = str(err.value)
    assert "courier_rating" in msg and "duplicate" in msg and "traffic" in msg


def test_validate_raw_missing_column(raw):
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_raw(raw.drop(columns=["weather"]))


def test_validate_table_catches_nulls(raw, small_cfg):
    table, _ = build_table(raw, small_cfg)
    validate_table(table, NUMERIC, CATEGORICAL, 50)
    table.loc[0, TARGET] = None
    with pytest.raises(DataValidationError, match="nulls"):
        validate_table(table, NUMERIC, CATEGORICAL, 50)
