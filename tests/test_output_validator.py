# tests/test_output_validator.py

"""
Tests for src/agent/validators/output_validator.py

All tests construct ExecutionResult objects inline — no subprocess needed.
"""

import numpy as np
import pandas as pd
import pytest

from src.agent.tools.code_executor import ExecutionResult, ExecutionStatus
from src.agent.validators.output_validator import (
    OutputValidator,
    ValidationFailureType,
    ValidationOutcome,
    _is_empty,
    _question_implies_plot,
    _question_implies_scalar,
    _question_implies_tabular,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _success(result) -> ExecutionResult:
    return ExecutionResult(status=ExecutionStatus.SUCCESS, result=result)


def _failure(status: ExecutionStatus, error_message: str = "error", retry_instruction: str = "retry") -> ExecutionResult:
    return ExecutionResult(
        status=status,
        error_message=error_message,
        retry_instruction=retry_instruction,
    )


VALIDATOR = OutputValidator()


# ---------------------------------------------------------------------------
# _is_empty helper
# ---------------------------------------------------------------------------

class TestIsEmpty:

    def test_none_is_empty(self):
        assert _is_empty(None) is True

    def test_empty_dataframe_is_empty(self):
        assert _is_empty(pd.DataFrame()) is True

    def test_nonempty_dataframe_is_not_empty(self):
        assert _is_empty(pd.DataFrame({"a": [1]})) is False

    def test_empty_list_is_empty(self):
        assert _is_empty([]) is True

    def test_nonempty_list_is_not_empty(self):
        assert _is_empty([1, 2, 3]) is False

    def test_empty_dict_is_empty(self):
        assert _is_empty({}) is True

    def test_nonempty_dict_is_not_empty(self):
        assert _is_empty({"a": 1}) is False

    def test_empty_string_is_empty(self):
        assert _is_empty("") is True
        assert _is_empty("   ") is True

    def test_nonempty_string_is_not_empty(self):
        assert _is_empty("hello") is False

    def test_zero_is_not_empty(self):
        # 0 is a valid scalar result — not empty
        assert _is_empty(0) is False

    def test_false_is_not_empty(self):
        assert _is_empty(False) is False

    def test_scalar_int_is_not_empty(self):
        assert _is_empty(42) is False

    def test_empty_series_is_empty(self):
        assert _is_empty(pd.Series([], dtype=float)) is True

    def test_nonempty_series_is_not_empty(self):
        assert _is_empty(pd.Series([1, 2, 3])) is False

    def test_empty_ndarray_is_empty(self):
        assert _is_empty(np.array([])) is True

    def test_nonempty_ndarray_is_not_empty(self):
        assert _is_empty(np.array([1, 2, 3])) is False


# ---------------------------------------------------------------------------
# Question signal helpers
# ---------------------------------------------------------------------------

class TestQuestionSignals:

    def test_scalar_signals(self):
        assert _question_implies_scalar("how many rows are there?") is True
        assert _question_implies_scalar("what is the total revenue?") is True
        assert _question_implies_scalar("what is the average score?") is True
        assert _question_implies_scalar("count the number of items") is True

    def test_tabular_signals(self):
        assert _question_implies_tabular("show me the top 10 customers") is True
        assert _question_implies_tabular("list all products in category A") is True
        assert _question_implies_tabular("which rows have null values?") is True
        assert _question_implies_tabular("filter rows where score > 90") is True

    def test_plot_signals(self):
        assert _question_implies_plot("plot the distribution of scores") is True
        assert _question_implies_plot("create a bar chart of sales by region") is True
        assert _question_implies_plot("show a histogram of ages") is True
        assert _question_implies_plot("visualise the trend over time") is True

    def test_no_signal_returns_false(self):
        assert _question_implies_scalar("do something with the data") is False
        assert _question_implies_tabular("do something with the data") is False
        assert _question_implies_plot("do something with the data") is False


# ---------------------------------------------------------------------------
# Validator — executor failure passthrough
# ---------------------------------------------------------------------------

class TestValidatorExecutorFailures:

    def test_runtime_error_fails(self):
        res = _failure(ExecutionStatus.RUNTIME_ERROR, "runtime error", "fix it")
        outcome = VALIDATOR.validate(res, "what is the total?")
        assert outcome.passed is False
        assert outcome.failure_type == ValidationFailureType.EXECUTION_ERROR

    def test_runtime_error_preserves_retry_instruction(self):
        res = _failure(ExecutionStatus.RUNTIME_ERROR, "err", "specific fix here")
        outcome = VALIDATOR.validate(res, "any question")
        assert outcome.retry_instruction == "specific fix here"

    def test_timeout_fails(self):
        res = _failure(ExecutionStatus.TIMEOUT, "timed out", "use vectorised ops")
        outcome = VALIDATOR.validate(res, "any question")
        assert outcome.passed is False
        assert outcome.failure_type == ValidationFailureType.TIMEOUT

    def test_timeout_preserves_retry_instruction(self):
        res = _failure(ExecutionStatus.TIMEOUT, "timed out", "use groupby instead")
        outcome = VALIDATOR.validate(res, "any question")
        assert outcome.retry_instruction == "use groupby instead"

    def test_forbidden_import_fails(self):
        res = _failure(ExecutionStatus.FORBIDDEN_IMPORT, "not permitted", "use pandas")
        outcome = VALIDATOR.validate(res, "any question")
        assert outcome.passed is False
        assert outcome.failure_type == ValidationFailureType.FORBIDDEN_IMPORT

    def test_missing_result_fails(self):
        res = _failure(ExecutionStatus.MISSING_RESULT, "no result", "assign result")
        outcome = VALIDATOR.validate(res, "any question")
        assert outcome.passed is False
        assert outcome.failure_type == ValidationFailureType.MISSING_RESULT

    def test_user_message_preserved_from_executor(self):
        res = _failure(ExecutionStatus.RUNTIME_ERROR, "column not found", "check schema")
        outcome = VALIDATOR.validate(res, "any question")
        assert outcome.user_message == "column not found"


# ---------------------------------------------------------------------------
# Validator — empty result
# ---------------------------------------------------------------------------

class TestValidatorEmptyResult:

    def test_none_result_fails(self):
        res = _success(None)
        outcome = VALIDATOR.validate(res, "how many rows?")
        assert outcome.passed is False
        assert outcome.failure_type == ValidationFailureType.EMPTY_RESULT

    def test_empty_dataframe_fails(self):
        res = _success(pd.DataFrame())
        outcome = VALIDATOR.validate(res, "show me the data")
        assert outcome.passed is False
        assert outcome.failure_type == ValidationFailureType.EMPTY_RESULT

    def test_empty_list_fails(self):
        res = _success([])
        outcome = VALIDATOR.validate(res, "list all items")
        assert outcome.passed is False
        assert outcome.failure_type == ValidationFailureType.EMPTY_RESULT

    def test_empty_string_fails(self):
        res = _success("   ")
        outcome = VALIDATOR.validate(res, "summarise the data")
        assert outcome.passed is False
        assert outcome.failure_type == ValidationFailureType.EMPTY_RESULT

    def test_empty_result_retry_mentions_column_names(self):
        res = _success(pd.DataFrame())
        outcome = VALIDATOR.validate(res, "show me filtered rows")
        assert "column" in outcome.retry_instruction.lower()

    def test_zero_scalar_passes(self):
        """0 is a valid answer (e.g. 'how many nulls?') — must not be flagged empty."""
        res = _success(0)
        outcome = VALIDATOR.validate(res, "how many null values are there?")
        assert outcome.passed is True

    def test_false_passes(self):
        res = _success(False)
        outcome = VALIDATOR.validate(res, "is any value missing?")
        assert outcome.passed is True


# ---------------------------------------------------------------------------
# Validator — implausible shape
# ---------------------------------------------------------------------------

class TestValidatorImplausibleShape:

    def test_scalar_question_with_large_dataframe_fails(self):
        q = "what is the total revenue?"
        assert _question_implies_scalar(q), (
            "Test question no longer triggers scalar signal — update the question"
        )
        df = pd.DataFrame({"a": range(10), "b": range(10)})
        res = _success(df)
        outcome = VALIDATOR.validate(res, q)
        assert outcome.passed is False
        assert outcome.failure_type == ValidationFailureType.IMPLAUSIBLE_SHAPE

    def test_scalar_question_with_single_row_df_passes(self):
        """A 1-row DataFrame is a plausible scalar-ish result.
        Depends on the scalar signal firing — the validator checks shape, not emptiness."""
        df = pd.DataFrame({"total": [1000]})
        res = _success(df)
        outcome = VALIDATOR.validate(res, "what is the total revenue?")
        assert outcome.passed is True

    def test_tabular_question_with_scalar_fails(self):
        q = "show me all rows where score > 80"
        assert _question_implies_tabular(q), (
            "Test question no longer triggers tabular signal — update the question"
        )
        res = _success(42)
        outcome = VALIDATOR.validate(res, q)
        assert outcome.passed is False
        assert outcome.failure_type == ValidationFailureType.IMPLAUSIBLE_SHAPE

    def test_tabular_question_with_dataframe_passes(self):
        """Depends on the tabular signal firing — validator checks shape but finds no mismatch."""
        df = pd.DataFrame({"name": ["Alice", "Bob"], "score": [85, 92]})
        res = _success(df)
        outcome = VALIDATOR.validate(res, "show me all rows where score > 80")
        assert outcome.passed is True

    def test_scalar_question_with_scalar_passes(self):
        """Depends on the scalar signal firing — validator checks shape but finds no mismatch."""
        res = _success(42)
        outcome = VALIDATOR.validate(res, "what is the average score?")
        assert outcome.passed is True

    def test_implausible_shape_retry_is_actionable(self):
        q = "what is the total count?"
        assert _question_implies_scalar(q), (
            "Test question no longer triggers scalar signal — update the question"
        )
        df = pd.DataFrame({"a": range(5)})
        res = _success(df)
        outcome = VALIDATOR.validate(res, q)
        assert len(outcome.retry_instruction) > 20
        assert "result" in outcome.retry_instruction.lower()

    def test_scalar_question_with_large_series_fails(self):
        q = "what is the average score?"
        assert _question_implies_scalar(q), (
            "Test question no longer triggers scalar signal — update the question"
        )
        res = _success(pd.Series([1, 2, 3, 4, 5]))
        outcome = VALIDATOR.validate(res, q)
        assert outcome.passed is False
        assert outcome.failure_type == ValidationFailureType.IMPLAUSIBLE_SHAPE

    def test_scalar_question_with_single_element_series_passes(self):
        """Depends on the scalar signal firing — a 1-element Series is plausible."""
        res = _success(pd.Series([42.0]))
        outcome = VALIDATOR.validate(res, "what is the average score?")
        assert outcome.passed is True

    def test_ambiguous_question_passes_with_any_result(self):
        """Questions with no clear signal should not trigger shape check."""
        q = "tell me about the data"
        assert not _question_implies_scalar(q), "Question unexpectedly triggers scalar signal"
        assert not _question_implies_tabular(q), "Question unexpectedly triggers tabular signal"
        assert not _question_implies_plot(q), "Question unexpectedly triggers plot signal"
        res = _success({"key": "value"})
        outcome = VALIDATOR.validate(res, q)
        assert outcome.passed is True


# ---------------------------------------------------------------------------
# Validator — SUCCESS paths
# ---------------------------------------------------------------------------

class TestValidatorSuccess:

    def test_scalar_result_passes(self):
        res = _success(42)
        outcome = VALIDATOR.validate(res, "what is the total count?")
        assert outcome.passed is True
        assert outcome.failure_type is None
        assert outcome.retry_instruction == ""
        assert outcome.user_message == ""

    def test_dataframe_result_passes(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        res = _success(df)
        outcome = VALIDATOR.validate(res, "show me the top rows")
        assert outcome.passed is True

    def test_dict_result_passes(self):
        res = _success({"mean": 85.0, "std": 5.2})
        outcome = VALIDATOR.validate(res, "give me summary statistics")
        assert outcome.passed is True

    def test_string_result_passes(self):
        res = _success("Alice")
        outcome = VALIDATOR.validate(res, "what is the name of the highest scorer?")
        assert outcome.passed is True

    def test_should_retry_false_on_success(self):
        res = _success(100)
        outcome = VALIDATOR.validate(res, "what is the count?")
        assert outcome.should_retry is False

    def test_should_retry_true_on_failure(self):
        res = _failure(ExecutionStatus.RUNTIME_ERROR)
        outcome = VALIDATOR.validate(res, "any question")
        assert outcome.should_retry is True


# ---------------------------------------------------------------------------
# ValidationOutcome dataclass
# ---------------------------------------------------------------------------

class TestValidationOutcome:

    def test_passed_outcome_defaults(self):
        outcome = ValidationOutcome(passed=True)
        assert outcome.failure_type is None
        assert outcome.user_message == ""
        assert outcome.retry_instruction == ""
        assert outcome.should_retry is False

    def test_failed_outcome_should_retry(self):
        outcome = ValidationOutcome(
            passed=False,
            failure_type=ValidationFailureType.EMPTY_RESULT,
            user_message="empty",
            retry_instruction="check filters",
        )
        assert outcome.should_retry is True

    def test_all_failure_types_are_distinct(self):
        types = list(ValidationFailureType)
        assert len(types) == len(set(types))