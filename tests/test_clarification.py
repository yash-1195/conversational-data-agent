"""
Tests for src/agent/clarification.py

All tests construct DataProfile objects inline — no file I/O needed.
"""

import pytest

from src.agent.clarification import (
    AmbiguityType,
    ClarificationResult,
    ClarificationTool,
    _check_column_ambiguity,
    _check_implied_grouping,
    _check_unbounded_time_range,
    _check_unspecified_aggregation,
)
from src.ingestion.profiler import ColumnProfile, DataProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(column_names: list[str], dtypes: dict[str, str] | None = None, flags: dict[str, list[str]] | None = None) -> DataProfile:
    """Build a minimal DataProfile with the given column names."""
    dtypes = dtypes or {}
    flags = flags or {}
    columns = [
        ColumnProfile(
            name=name,
            dtype=dtypes.get(name, "object"),
            null_count=0,
            null_rate=0.0,
            unique_count=10,
            sample_values=["a", "b", "c"],
            flags=flags.get(name, []),
        )
        for name in column_names
    ]
    return DataProfile(
        dataset_name="test.csv",
        row_count=100,
        col_count=len(columns),
        columns=columns,
    )


TOOL = ClarificationTool()


# ---------------------------------------------------------------------------
# _check_column_ambiguity
# ---------------------------------------------------------------------------

class TestColumnAmbiguity:

    def test_no_ambiguity_when_one_column_matches(self):
        profile = _make_profile(["revenue", "region", "date"])
        result = _check_column_ambiguity("what is the total revenue?", profile)
        assert result is None

    def test_ambiguity_when_two_columns_match(self):
        profile = _make_profile(["revenue", "region", "date"])
        result = _check_column_ambiguity("compare revenue and region", profile)
        assert result is not None
        assert result.ambiguity_type == AmbiguityType.COLUMN_AMBIGUITY

    def test_ambiguity_clarifying_question_names_columns(self):
        profile = _make_profile(["sales", "salary", "date"])
        result = _check_column_ambiguity("show me the sales and salary data", profile)
        assert result is not None
        assert "sales" in result.clarifying_question
        assert "salary" in result.clarifying_question

    def test_no_match_when_no_column_in_question(self):
        profile = _make_profile(["revenue", "region", "date"])
        result = _check_column_ambiguity("how many rows are there?", profile)
        assert result is None

    def test_short_column_names_below_threshold_not_matched(self):
        # Columns shorter than _MIN_COLUMN_MATCH_CHARS should not trigger
        profile = _make_profile(["id", "revenue", "date"])
        result = _check_column_ambiguity("what is the revenue by id?", profile)
        # "id" is 2 chars — below threshold — so only revenue matches → no ambiguity
        assert result is None

    def test_three_column_matches_still_fires(self):
        profile = _make_profile(["sales", "salary", "savings"])
        result = _check_column_ambiguity("compare sales salary and savings", profile)
        assert result is not None
        assert result.ambiguity_type == AmbiguityType.COLUMN_AMBIGUITY

    def test_returns_none_for_empty_profile(self):
        profile = _make_profile([])
        result = _check_column_ambiguity("what is the total?", profile)
        assert result is None

    def test_column_ambiguity_suppressed_when_columns_flank_grouping_trigger(self):
        """'revenue by region' is unambiguous — column roles are clear."""
        profile = _make_profile(["revenue", "region"])
        result = _check_column_ambiguity("show me revenue by region", profile)
        assert result is None


# ---------------------------------------------------------------------------
# _check_unspecified_aggregation
# ---------------------------------------------------------------------------

class TestUnspecifiedAggregation:

    def test_fires_on_average(self):
        result = _check_unspecified_aggregation("what is the average sales?")
        assert result is not None
        assert result.ambiguity_type == AmbiguityType.UNSPECIFIED_AGG

    def test_fires_on_typical(self):
        result = _check_unspecified_aggregation("what is the typical order value?")
        assert result is not None

    def test_fires_on_representative(self):
        result = _check_unspecified_aggregation("give me the representative score")
        assert result is not None

    def test_does_not_fire_when_mean_specified(self):
        result = _check_unspecified_aggregation("what is the mean sales?")
        assert result is None

    def test_does_not_fire_when_median_specified(self):
        result = _check_unspecified_aggregation("what is the median order value?")
        assert result is None

    def test_does_not_fire_when_sum_specified(self):
        result = _check_unspecified_aggregation("what is the total sum of revenue?")
        assert result is None

    def test_does_not_fire_on_unrelated_question(self):
        result = _check_unspecified_aggregation("show me all rows where score > 80")
        assert result is None

    def test_clarifying_question_mentions_triggered_word(self):
        result = _check_unspecified_aggregation("what is the average score?")
        assert result is not None
        assert "average" in result.clarifying_question

    def test_clarifying_question_offers_options(self):
        result = _check_unspecified_aggregation("what is the typical revenue?")
        assert result is not None
        # Should offer concrete aggregation options
        assert any(
            word in result.clarifying_question.lower()
            for word in ["mean", "median", "mode"]
        )

    def test_does_not_suppress_on_count_as_substring(self):
        # "account" and "discount" contain "count" as a substring —
        # must not suppress the aggregation check (only whole-word matches count)
        result = _check_unspecified_aggregation("what is the average account discount?")
        assert result is not None
        assert result.ambiguity_type == AmbiguityType.UNSPECIFIED_AGG


# ---------------------------------------------------------------------------
# _check_unbounded_time_range
# ---------------------------------------------------------------------------

class TestUnboundedTimeRange:

    def test_fires_on_recently_with_date_column(self):
        profile = _make_profile(["revenue", "date"])
        result = _check_unbounded_time_range("show me recent sales", profile)
        assert result is not None
        assert result.ambiguity_type == AmbiguityType.UNBOUNDED_TIME_RANGE

    def test_fires_on_lately_with_date_column(self):
        profile = _make_profile(["revenue", "date"])
        result = _check_unbounded_time_range("what has been selling lately?", profile)
        assert result is not None

    def test_does_not_fire_without_date_column(self):
        profile = _make_profile(["revenue", "region"])
        result = _check_unbounded_time_range("show me recent sales", profile)
        assert result is None

    def test_does_not_fire_when_year_anchor_present(self):
        profile = _make_profile(["revenue", "date"])
        result = _check_unbounded_time_range("show me recent sales from 2024", profile)
        assert result is None

    def test_does_not_fire_when_month_anchor_present(self):
        profile = _make_profile(["revenue", "date"])
        result = _check_unbounded_time_range("show me sales from january", profile)
        assert result is None

    def test_does_not_fire_when_since_anchor_present(self):
        profile = _make_profile(["revenue", "date"])
        result = _check_unbounded_time_range("show me sales since last quarter", profile)
        assert result is None

    def test_fires_with_datetime_dtype_column(self):
        profile = _make_profile(
            ["revenue", "order_date"],
            dtypes={"order_date": "datetime64[ns]", "revenue": "float64"},
        )
        result = _check_unbounded_time_range("show me the latest orders", profile)
        assert result is not None

    def test_fires_with_date_flag_on_column(self):
        profile = _make_profile(
            ["revenue", "created_at"],
            flags={"created_at": ["possible date column stored as string"]},
        )
        result = _check_unbounded_time_range("show me recent revenue", profile)
        assert result is not None

    def test_clarifying_question_mentions_trigger_word(self):
        profile = _make_profile(["revenue", "date"])
        result = _check_unbounded_time_range("show me the latest revenue", profile)
        assert result is not None
        assert "latest" in result.clarifying_question

    def test_does_not_suppress_on_from_in_nontemporal_context(self):
        # "from" in a non-temporal phrase ("from the north region") must not
        # act as a time anchor and suppress the clarification check.
        profile = _make_profile(["revenue", "date"])
        result = _check_unbounded_time_range(
            "show me recent revenue from the north region", profile
        )
        assert result is not None

    def test_does_not_suppress_on_may_as_auxiliary_verb(self):
        # "may" as an auxiliary verb must not suppress the time range check.
        profile = _make_profile(["revenue", "date"])
        result = _check_unbounded_time_range(
            "show me recent trends that may continue", profile
        )
        assert result is not None


# ---------------------------------------------------------------------------
# _check_implied_grouping
# ---------------------------------------------------------------------------

class TestImpliedGrouping:

    def test_fires_when_by_used_without_column(self):
        profile = _make_profile(["revenue", "region", "date"])
        result = _check_implied_grouping("show me revenue by category", profile)
        assert result is not None
        assert result.ambiguity_type == AmbiguityType.IMPLIED_GROUPING

    def test_does_not_fire_when_column_named_after_by(self):
        profile = _make_profile(["revenue", "region", "date"])
        result = _check_implied_grouping("show me revenue by region", profile)
        assert result is None

    def test_fires_on_per(self):
        profile = _make_profile(["revenue", "region"])
        result = _check_implied_grouping("what is the revenue per segment?", profile)
        assert result is not None

    def test_does_not_fire_on_per_with_known_column(self):
        profile = _make_profile(["revenue", "region"])
        result = _check_implied_grouping("what is the revenue per region?", profile)
        assert result is None

    def test_fires_on_for_each(self):
        profile = _make_profile(["revenue", "region"])
        result = _check_implied_grouping("show me revenue for each segment", profile)
        assert result is not None

    def test_does_not_fire_without_grouping_trigger(self):
        profile = _make_profile(["revenue", "region"])
        result = _check_implied_grouping("what is the total revenue?", profile)
        assert result is None

    def test_clarifying_question_lists_available_columns(self):
        profile = _make_profile(["revenue", "region", "category"])
        result = _check_implied_grouping("show me revenue by segment", profile)
        assert result is not None
        assert "revenue" in result.clarifying_question
        assert "region" in result.clarifying_question
        assert "category" in result.clarifying_question

    def test_fires_on_broken_down_by(self):
        profile = _make_profile(["revenue", "region"])
        result = _check_implied_grouping("show me revenue broken down by segment", profile)
        assert result is not None


# ---------------------------------------------------------------------------
# ClarificationTool.check — priority ordering
# ---------------------------------------------------------------------------

class TestClarificationToolPriority:

    def test_no_clarification_needed_for_clear_question(self):
        profile = _make_profile(["revenue", "region", "date"])
        result = TOOL.check("what is the total revenue by region?", profile)
        assert result.needs_clarification is False
        assert result.ambiguity_type is None
        assert result.clarifying_question == ""

    def test_column_ambiguity_fires_first(self):
        """Column ambiguity should take priority over aggregation ambiguity."""
        profile = _make_profile(["sales", "salary"])
        # "average" triggers agg check, "sales"+"salary" triggers column check
        result = TOOL.check("what is the average sales and salary?", profile)
        assert result.needs_clarification is True
        assert result.ambiguity_type == AmbiguityType.COLUMN_AMBIGUITY

    def test_agg_fires_when_no_column_ambiguity(self):
        profile = _make_profile(["revenue", "region"])
        result = TOOL.check("what is the typical revenue?", profile)
        assert result.needs_clarification is True
        assert result.ambiguity_type == AmbiguityType.UNSPECIFIED_AGG

    def test_time_range_fires_when_no_higher_priority_match(self):
        profile = _make_profile(["revenue", "date"])
        result = TOOL.check("show me recent revenue", profile)
        assert result.needs_clarification is True
        assert result.ambiguity_type == AmbiguityType.UNBOUNDED_TIME_RANGE

    def test_grouping_fires_when_no_higher_priority_match(self):
        profile = _make_profile(["revenue", "region"])
        result = TOOL.check("show me revenue by segment", profile)
        assert result.needs_clarification is True
        assert result.ambiguity_type == AmbiguityType.IMPLIED_GROUPING

    def test_returns_false_when_all_heuristics_pass(self):
        profile = _make_profile(["revenue", "region", "date"])
        result = TOOL.check("what is the mean revenue by region in 2024?", profile)
        assert result.needs_clarification is False


# ---------------------------------------------------------------------------
# ClarificationResult dataclass
# ---------------------------------------------------------------------------

class TestClarificationResult:

    def test_passing_result_defaults(self):
        result = ClarificationResult(needs_clarification=False)
        assert result.ambiguity_type is None
        assert result.clarifying_question == ""
        assert result.detail == ""

    def test_all_ambiguity_types_are_distinct(self):
        types = list(AmbiguityType)
        assert len(types) == len(set(types))

    def test_failing_result_has_clarifying_question(self):
        result = ClarificationResult(
            needs_clarification=True,
            ambiguity_type=AmbiguityType.UNSPECIFIED_AGG,
            clarifying_question="Which aggregation?",
            detail="triggered by: average",
        )
        assert result.needs_clarification is True
        assert result.clarifying_question == "Which aggregation?"