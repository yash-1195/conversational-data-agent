"""
Tests for OutputFormatter.

All tests are self-contained — DataFrames constructed inline, plots created
with matplotlib directly. No external files or fixtures required.

Coverage
--------
- DataFrame result: answer_type, content format, truncation at _TABLE_MAX_ROWS
- Series result: promoted to DataFrame, answer_type = "table"
- Scalar types: int, float, bool, str → answer_type = "text"
- numpy scalar: extracted to Python value, no "np.int64(...)" in output
- Unsupported type (list, dict): str() fallback, answer_type = "text"
- matplotlib Figure: answer_type = "plot", plot_path set, PNG file exists
- Caveat — column-level flags: only flagged columns present in result surfaced
- Caveat — column not in result: flag not surfaced even if profiler flagged it
- Caveat — dataset-level flags: surfaced for all result types
- Caveat — clean profile: empty caveat string
- Caveat — text result: only dataset-level flags, no column flags
"""

import os
import numpy as np
import pandas as pd
import pytest
import matplotlib
import matplotlib.figure
matplotlib.use("Agg")  # headless — no display required
import matplotlib.pyplot as plt

from src.agent.output_formatter import OutputFormatter, _TABLE_MAX_ROWS
from src.ingestion.profiler import ColumnProfile, DataProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(
    col_flags: dict[str, list[str]] | None = None,
    dataset_flags: list[str] | None = None,
) -> DataProfile:
    """
    Build a minimal DataProfile for testing.

    Parameters
    ----------
    col_flags:
        Mapping of column name → list of flag strings. Columns listed here
        are included in the profile with those flags set.
    dataset_flags:
        Dataset-level quality flag strings.
    """
    col_flags = col_flags or {}
    dataset_flags = dataset_flags or []

    columns = [
        ColumnProfile(
            name=name,
            dtype="object",
            null_count=0,
            null_rate=0.0,
            unique_count=1,
            sample_values=["x"],
            flags=flags,
        )
        for name, flags in col_flags.items()
    ]

    return DataProfile(
        dataset_name="test.csv",
        row_count=10,
        col_count=len(columns),
        columns=columns,
        quality_flags=dataset_flags,
    )


def _clean_profile() -> DataProfile:
    """Profile with no flags at all."""
    return _make_profile()


def _make_figure() -> matplotlib.figure.Figure:
    """Create a minimal matplotlib Figure."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6])
    return fig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def formatter() -> OutputFormatter:
    return OutputFormatter()


@pytest.fixture
def tmp_plot_dir(tmp_path) -> str:
    return str(tmp_path / "plots")


# ---------------------------------------------------------------------------
# DataFrame results
# ---------------------------------------------------------------------------

class TestDataFrameResult:
    def test_answer_type_is_table(self, formatter, tmp_plot_dir):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        output = formatter.format(df, _clean_profile(), tmp_plot_dir)
        assert output.answer_type == "table"

    def test_plot_path_is_none(self, formatter, tmp_plot_dir):
        df = pd.DataFrame({"a": [1]})
        output = formatter.format(df, _clean_profile(), tmp_plot_dir)
        assert output.plot_path is None

    def test_content_contains_column_names(self, formatter, tmp_plot_dir):
        df = pd.DataFrame({"revenue": [100], "region": ["North"]})
        output = formatter.format(df, _clean_profile(), tmp_plot_dir)
        assert "revenue" in output.content
        assert "region" in output.content

    def test_no_truncation_below_limit(self, formatter, tmp_plot_dir):
        df = pd.DataFrame({"x": range(_TABLE_MAX_ROWS)})
        output = formatter.format(df, _clean_profile(), tmp_plot_dir)
        assert "Showing first" not in output.content

    def test_truncation_notice_above_limit(self, formatter, tmp_plot_dir):
        df = pd.DataFrame({"x": range(_TABLE_MAX_ROWS + 1)})
        output = formatter.format(df, _clean_profile(), tmp_plot_dir)
        assert "Showing first" in output.content
        assert str(_TABLE_MAX_ROWS) in output.content


# ---------------------------------------------------------------------------
# Series result
# ---------------------------------------------------------------------------

class TestSeriesResult:
    def test_series_promoted_to_table(self, formatter, tmp_plot_dir):
        s = pd.Series([10, 20, 30], name="sales")
        output = formatter.format(s, _clean_profile(), tmp_plot_dir)
        assert output.answer_type == "table"

    def test_series_content_contains_values(self, formatter, tmp_plot_dir):
        s = pd.Series([10, 20], name="count")
        output = formatter.format(s, _clean_profile(), tmp_plot_dir)
        assert "10" in output.content
        assert "20" in output.content


# ---------------------------------------------------------------------------
# Scalar results
# ---------------------------------------------------------------------------

class TestScalarResult:
    def test_int(self, formatter, tmp_plot_dir):
        output = formatter.format(42, _clean_profile(), tmp_plot_dir)
        assert output.answer_type == "text"
        assert output.content == "42"

    def test_float(self, formatter, tmp_plot_dir):
        output = formatter.format(3.14, _clean_profile(), tmp_plot_dir)
        assert output.answer_type == "text"
        assert "3.14" in output.content

    def test_bool(self, formatter, tmp_plot_dir):
        output = formatter.format(True, _clean_profile(), tmp_plot_dir)
        assert output.answer_type == "text"
        assert output.content == "True"

    def test_string(self, formatter, tmp_plot_dir):
        output = formatter.format("hello", _clean_profile(), tmp_plot_dir)
        assert output.answer_type == "text"
        assert output.content == "hello"

    def test_numpy_int_no_type_wrapper(self, formatter, tmp_plot_dir):
        value = np.int64(99)
        output = formatter.format(value, _clean_profile(), tmp_plot_dir)
        assert output.answer_type == "text"
        assert output.content == "99"
        # Ensure numpy type repr is not leaked into the output
        assert "int64" not in output.content
        assert "np." not in output.content

    def test_numpy_float_no_type_wrapper(self, formatter, tmp_plot_dir):
        value = np.float64(1.5)
        output = formatter.format(value, _clean_profile(), tmp_plot_dir)
        assert output.answer_type == "text"
        assert "float64" not in output.content


# ---------------------------------------------------------------------------
# Fallback result types
# ---------------------------------------------------------------------------

class TestFallbackResult:
    def test_list_falls_back_to_text(self, formatter, tmp_plot_dir):
        output = formatter.format([1, 2, 3], _clean_profile(), tmp_plot_dir)
        assert output.answer_type == "text"
        assert "1" in output.content

    def test_dict_falls_back_to_text(self, formatter, tmp_plot_dir):
        output = formatter.format({"a": 1}, _clean_profile(), tmp_plot_dir)
        assert output.answer_type == "text"


# ---------------------------------------------------------------------------
# matplotlib Figure results
# ---------------------------------------------------------------------------

class TestMatplotlibResult:
    def test_answer_type_is_plot(self, formatter, tmp_plot_dir):
        fig = _make_figure()
        output = formatter.format(fig, _clean_profile(), tmp_plot_dir)
        assert output.answer_type == "plot"

    def test_plot_path_is_set(self, formatter, tmp_plot_dir):
        fig = _make_figure()
        output = formatter.format(fig, _clean_profile(), tmp_plot_dir)
        assert output.plot_path is not None

    def test_content_equals_plot_path(self, formatter, tmp_plot_dir):
        fig = _make_figure()
        output = formatter.format(fig, _clean_profile(), tmp_plot_dir)
        assert output.content == output.plot_path

    def test_png_file_exists_on_disk(self, formatter, tmp_plot_dir):
        fig = _make_figure()
        output = formatter.format(fig, _clean_profile(), tmp_plot_dir)
        assert os.path.isfile(output.plot_path)

    def test_saved_file_is_png(self, formatter, tmp_plot_dir):
        fig = _make_figure()
        output = formatter.format(fig, _clean_profile(), tmp_plot_dir)
        assert output.plot_path.endswith(".png")

    def test_plot_dir_created_if_missing(self, formatter, tmp_path):
        plot_dir = str(tmp_path / "new" / "nested" / "dir")
        assert not os.path.exists(plot_dir)
        fig = _make_figure()
        formatter.format(fig, _clean_profile(), plot_dir)
        assert os.path.isdir(plot_dir)

    def test_multiple_plots_get_unique_paths(self, formatter, tmp_plot_dir):
        fig1 = _make_figure()
        fig2 = _make_figure()
        out1 = formatter.format(fig1, _clean_profile(), tmp_plot_dir)
        out2 = formatter.format(fig2, _clean_profile(), tmp_plot_dir)
        assert out1.plot_path != out2.plot_path


# ---------------------------------------------------------------------------
# Caveat — column-level flags
# ---------------------------------------------------------------------------

class TestColumnCaveat:
    def test_flagged_column_in_result_surfaces_caveat(self, formatter, tmp_plot_dir):
        df = pd.DataFrame({"revenue": [1, 2], "region": ["A", "B"]})
        profile = _make_profile(
            col_flags={"revenue": ["high null rate (45%)"]},
        )
        output = formatter.format(df, profile, tmp_plot_dir)
        assert "revenue" in output.caveat
        assert "high null rate" in output.caveat

    def test_flagged_column_not_in_result_not_surfaced(self, formatter, tmp_plot_dir):
        # 'cost' is flagged but not in the result DataFrame
        df = pd.DataFrame({"revenue": [1, 2]})
        profile = _make_profile(
            col_flags={"cost": ["high null rate (50%)"]},
        )
        output = formatter.format(df, profile, tmp_plot_dir)
        assert "cost" not in output.caveat

    def test_multiple_flagged_columns_all_surfaced(self, formatter, tmp_plot_dir):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        profile = _make_profile(
            col_flags={
                "a": ["high null rate (30%)"],
                "b": ["mixed types detected"],
            },
        )
        output = formatter.format(df, profile, tmp_plot_dir)
        assert "a" in output.caveat
        assert "b" in output.caveat

    def test_clean_profile_empty_caveat_for_dataframe(self, formatter, tmp_plot_dir):
        df = pd.DataFrame({"x": [1, 2]})
        output = formatter.format(df, _clean_profile(), tmp_plot_dir)
        assert output.caveat == ""


# ---------------------------------------------------------------------------
# Caveat — dataset-level flags
# ---------------------------------------------------------------------------

class TestDatasetLevelCaveat:
    def test_dataset_flag_surfaced_for_dataframe(self, formatter, tmp_plot_dir):
        df = pd.DataFrame({"x": [1]})
        profile = _make_profile(dataset_flags=["1,000 duplicate rows (10.0%)"])
        output = formatter.format(df, profile, tmp_plot_dir)
        assert "duplicate rows" in output.caveat

    def test_dataset_flag_surfaced_for_scalar(self, formatter, tmp_plot_dir):
        profile = _make_profile(dataset_flags=["entirely null columns: notes"])
        output = formatter.format(42, profile, tmp_plot_dir)
        assert "entirely null columns" in output.caveat

    def test_dataset_flag_surfaced_for_plot(self, formatter, tmp_plot_dir):
        profile = _make_profile(dataset_flags=["500 duplicate rows (5.0%)"])
        fig = _make_figure()
        output = formatter.format(fig, profile, tmp_plot_dir)
        assert "duplicate rows" in output.caveat

    def test_clean_profile_empty_caveat_for_scalar(self, formatter, tmp_plot_dir):
        output = formatter.format(99, _clean_profile(), tmp_plot_dir)
        assert output.caveat == ""

    def test_clean_profile_empty_caveat_for_plot(self, formatter, tmp_plot_dir):
        fig = _make_figure()
        output = formatter.format(fig, _clean_profile(), tmp_plot_dir)
        assert output.caveat == ""

    def test_caveat_header_present_when_flagged(self, formatter, tmp_plot_dir):
        profile = _make_profile(dataset_flags=["some flag"])
        output = formatter.format(42, profile, tmp_plot_dir)
        assert "⚠" in output.caveat