"""Tests for src/ingestion/profiler.py"""

import pandas as pd
import pytest

from src.ingestion.profiler import DataProfiler, DataProfile, ColumnProfile


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _profiler() -> DataProfiler:
    return DataProfiler()


def _profile(df: pd.DataFrame, name: str = "test.csv") -> DataProfile:
    return _profiler().profile(df, dataset_name=name)


# ------------------------------------------------------------------
# Basic structure
# ------------------------------------------------------------------

class TestDataProfileStructure:
    def test_returns_dataprofile(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        p = _profile(df)
        assert isinstance(p, DataProfile)

    def test_row_and_col_counts(self):
        df = pd.DataFrame({"x": range(10), "y": range(10), "z": range(10)})
        p = _profile(df)
        assert p.row_count == 10
        assert p.col_count == 3

    def test_dataset_name_preserved(self):
        df = pd.DataFrame({"a": [1]})
        p = _profile(df, name="sales_q1.csv")
        assert p.dataset_name == "sales_q1.csv"

    def test_column_profiles_match_columns(self):
        df = pd.DataFrame({"a": [1], "b": ["x"], "c": [1.5]})
        p = _profile(df)
        assert len(p.columns) == 3
        assert [c.name for c in p.columns] == ["a", "b", "c"]


# ------------------------------------------------------------------
# Null rate detection
# ------------------------------------------------------------------

class TestNullRateFlag:
    def test_high_null_rate_flagged(self):
        df = pd.DataFrame({"col": [None] * 80 + [1] * 20})
        p = _profile(df)
        col = p.columns[0]
        assert any("high null rate" in f for f in col.flags)

    def test_low_null_rate_not_flagged(self):
        df = pd.DataFrame({"col": [None] + [1] * 99})
        p = _profile(df)
        col = p.columns[0]
        assert not any("high null rate" in f for f in col.flags)

    def test_null_count_and_rate_correct(self):
        df = pd.DataFrame({"col": [None, None, 1, 2, 3, 4, 5, 6, 7, 8]})
        p = _profile(df)
        col = p.columns[0]
        assert col.null_count == 2
        assert abs(col.null_rate - 0.2) < 0.001

    def test_zero_nulls(self):
        df = pd.DataFrame({"col": [1, 2, 3]})
        p = _profile(df)
        assert p.columns[0].null_count == 0
        assert p.columns[0].null_rate == 0.0


# ------------------------------------------------------------------
# Date string detection
# ------------------------------------------------------------------

class TestDateStringFlag:
    def test_iso_date_strings_flagged(self):
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-02-15", "2024-03-20"] * 10
        })
        p = _profile(df)
        flags = p.columns[0].flags
        assert any("date" in f for f in flags)

    def test_us_date_strings_flagged(self):
        df = pd.DataFrame({
            "date": ["01/15/2024", "02/20/2024", "12/31/2023"] * 10
        })
        p = _profile(df)
        flags = p.columns[0].flags
        assert any("date" in f for f in flags)

    def test_actual_datetime_column_not_flagged(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-02-01"])})
        p = _profile(df)
        # datetime64 dtype — not object, so should not trigger date string flag
        flags = p.columns[0].flags
        assert not any("date column stored as string" in f for f in flags)

    def test_non_date_strings_not_flagged(self):
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"] * 10})
        p = _profile(df)
        flags = p.columns[0].flags
        assert not any("date" in f for f in flags)


# ------------------------------------------------------------------
# Mixed type detection
# ------------------------------------------------------------------

class TestMixedTypeFlag:
    def test_mixed_numeric_and_string_flagged(self):
        # 60% numeric, 40% string — should flag
        values = [1, 2, 3, 4, 5, 6, "foo", "bar", "baz", "qux"]
        df = pd.DataFrame({"col": values * 20})
        p = _profile(df)
        assert any("mixed types" in f for f in p.columns[0].flags)

    def test_fully_numeric_string_not_flagged(self):
        df = pd.DataFrame({"col": ["1", "2", "3", "4"] * 25})
        p = _profile(df)
        # All parse as numeric — not mixed
        assert not any("mixed types" in f for f in p.columns[0].flags)

    def test_fully_text_not_flagged(self):
        df = pd.DataFrame({"col": ["foo", "bar", "baz"] * 33})
        p = _profile(df)
        assert not any("mixed types" in f for f in p.columns[0].flags)

    def test_integer_column_not_flagged(self):
        df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
        p = _profile(df)
        assert not any("mixed types" in f for f in p.columns[0].flags)


# ------------------------------------------------------------------
# Dataset-level flags
# ------------------------------------------------------------------

class TestDatasetFlags:
    def test_duplicate_rows_flagged(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        p = _profile(df)
        assert any("duplicate" in f for f in p.quality_flags)

    def test_no_duplicates_not_flagged(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        p = _profile(df)
        assert not any("duplicate" in f for f in p.quality_flags)

    def test_entirely_null_column_flagged(self):
        df = pd.DataFrame({"a": [1, 2, 3], "empty": [None, None, None]})
        p = _profile(df)
        assert any("entirely null" in f for f in p.quality_flags)

    def test_clean_dataset_no_flags(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        p = _profile(df)
        assert p.quality_flags == []


# ------------------------------------------------------------------
# Prompt rendering
# ------------------------------------------------------------------

class TestPromptRendering:
    def test_to_prompt_str_includes_dataset_name(self):
        df = pd.DataFrame({"a": [1, 2]})
        p = _profile(df, name="my_data.csv")
        assert "my_data.csv" in p.to_prompt_str()

    def test_to_prompt_str_includes_shape(self):
        df = pd.DataFrame({"a": range(50), "b": range(50)})
        p = _profile(df)
        prompt = p.to_prompt_str()
        assert "50" in prompt
        assert "2" in prompt

    def test_to_prompt_str_includes_column_names(self):
        df = pd.DataFrame({"revenue": [1], "region": ["EU"]})
        p = _profile(df)
        prompt = p.to_prompt_str()
        assert "revenue" in prompt
        assert "region" in prompt

    def test_to_prompt_str_includes_flags(self):
        df = pd.DataFrame({"col": [None] * 80 + [1] * 20})
        p = _profile(df)
        prompt = p.to_prompt_str()
        assert "FLAG" in prompt.upper()

    def test_to_prompt_str_returns_string(self):
        df = pd.DataFrame({"a": [1]})
        assert isinstance(_profile(df).to_prompt_str(), str)


# ------------------------------------------------------------------
# Helper methods
# ------------------------------------------------------------------

class TestProfileHelpers:
    def test_flagged_columns_returns_names(self):
        df = pd.DataFrame({
            "clean": list(range(100)),
            "nulls": [None] * 80 + list(range(20)),
        })
        p = _profile(df)
        flagged = p.flagged_columns()
        assert "nulls" in flagged
        assert "clean" not in flagged

    def test_get_column_flags_for_existing_column(self):
        df = pd.DataFrame({"col": [None] * 80 + [1] * 20})
        p = _profile(df)
        flags = p.get_column_flags("col")
        assert isinstance(flags, list)
        assert len(flags) > 0

    def test_get_column_flags_for_missing_column(self):
        df = pd.DataFrame({"a": [1]})
        p = _profile(df)
        assert p.get_column_flags("nonexistent") == []

    def test_sample_values_capped(self):
        df = pd.DataFrame({"col": range(1000)})
        p = _profile(df, name="d.csv")
        # sample_rows default is 5
        assert len(p.columns[0].sample_values) <= 5