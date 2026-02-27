# src/agent/validators/output_validator.py

"""
OutputValidator — structured checklist that runs inside the `evaluate` node.

One user question + one ExecutionResult → one ValidationOutcome.
Each failure type maps to a distinct retry instruction so the `plan` node
can self-correct intelligently rather than just re-running the same code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd
import numpy as np

from src.agent.tools.code_executor import ExecutionResult, ExecutionStatus

try:
    import matplotlib.figure as _mpl_figure
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Failure taxonomy
# ---------------------------------------------------------------------------

class ValidationFailureType(Enum):
    # Failures that originate in the executor
    EXECUTION_ERROR   = "execution_error"    # runtime error
    TIMEOUT           = "timeout"            # sandbox timed out
    FORBIDDEN_IMPORT  = "forbidden_import"   # blocked module / dangerous call
    MISSING_RESULT    = "missing_result"     # `result` variable never assigned

    # Failures detected by the validator after successful execution
    EMPTY_RESULT      = "empty_result"       # result is None / empty df / empty list
    IMPLAUSIBLE_SHAPE = "implausible_shape"  # result type doesn't fit the question


# ---------------------------------------------------------------------------
# Outcome dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationOutcome:
    passed: bool

    # Populated on failure only
    failure_type: ValidationFailureType | None = None

    # Safe, user-facing explanation of what went wrong
    user_message: str = ""

    # Appended to the LLM prompt on the next plan attempt
    retry_instruction: str = ""

    @property
    def should_retry(self) -> bool:
        """
        True for all failures — the graph uses this to decide whether to
        loop back to `plan` or route to `respond`.
        """
        return not self.passed


# ---------------------------------------------------------------------------
# Shape heuristics
# ---------------------------------------------------------------------------

# Keywords that strongly imply the answer should be a single number or string
_SCALAR_SIGNALS = frozenset({
    "how many", "how much", "what is the total", "what is the average",
    "what is the mean", "what is the median", "what is the max",
    "what is the min", "what is the sum", "count", "total", "average",
    "mean", "median", "maximum", "minimum", "percentage", "percent",
    "ratio", "rate", "what is the",
})

# Keywords that imply the answer should be tabular (DataFrame / list)
_TABULAR_SIGNALS = frozenset({
    "show me", "list", "display", "which", "top", "bottom",
    "rows", "records", "entries", "show all", "give me all",
    "find all", "filter", "where", "what are the",
})

# Keywords that imply the answer should be a plot / figure
_PLOT_SIGNALS = frozenset({
    "plot", "chart", "graph", "visuali", "histogram", "bar chart",
    "line chart", "scatter", "pie chart", "heatmap", "distribution",
})


def _question_implies_scalar(question: str) -> bool:
    q = question.lower()
    return any(signal in q for signal in _SCALAR_SIGNALS)


def _question_implies_tabular(question: str) -> bool:
    q = question.lower()
    return any(signal in q for signal in _TABULAR_SIGNALS)


def _question_implies_plot(question: str) -> bool:
    q = question.lower()
    return any(signal in q for signal in _PLOT_SIGNALS)


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (int, float, str, bool)) or (
        hasattr(value, "item")  # numpy scalar
    )


def _is_empty(value: Any) -> bool:
    """Return True if the result is None or has zero elements."""
    if value is None:
        return True
    if isinstance(value, pd.DataFrame):
        return value.empty
    if isinstance(value, pd.Series):
        return value.empty
    if isinstance(value, np.ndarray):
        return value.size == 0
    if isinstance(value, (list, dict, set, tuple)):
        return len(value) == 0
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _check_shape_plausibility(result: Any, question: str) -> str | None:
    """
    Lightweight heuristic check — returns a human-readable mismatch
    description if something seems wrong, or None if plausible.

    Deliberately lenient: only flags clear mismatches to avoid false
    positives on ambiguous questions.
    """
    is_df      = isinstance(result, pd.DataFrame)
    is_series  = isinstance(result, pd.Series)
    is_scalar_ = _is_scalar(result)

    # Question strongly implies a plot but result is not a figure
    if _question_implies_plot(question):
        if _MATPLOTLIB_AVAILABLE:
            if not isinstance(result, _mpl_figure.Figure):
                return (
                    "The question asks for a chart or plot, but `result` is not a "
                    f"matplotlib Figure (got {type(result).__name__}). "
                    "Create a matplotlib figure and assign it to `result`."
                )
        elif _is_scalar(result) or isinstance(result, (pd.DataFrame, pd.Series)):
            # matplotlib not installed — still flag obviously wrong result types
            return (
                "The question asks for a chart or plot, but `result` is a "
                f"{type(result).__name__}. Create a matplotlib figure and assign "
                "it to `result` (e.g. `fig, ax = plt.subplots(); ...; result = fig`)."
            )

    # Question strongly implies a scalar but result is a non-trivial DataFrame or Series
    if _question_implies_scalar(question):
        if is_df and len(result) > 1:
            return (
                "The question asks for a single value, but `result` is a DataFrame "
                f"with {len(result)} rows. Reduce it to a scalar "
                "(e.g. `.sum()`, `.mean()`, `.iloc[0]`)."
            )
        if is_series and len(result) > 1:
            return (
                "The question asks for a single value, but `result` is a Series "
                f"with {len(result)} elements. Reduce it to a scalar "
                "(e.g. `.sum()`, `.mean()`, `.iloc[0]`)."
            )

    # Question strongly implies tabular output but result is a scalar
    if _question_implies_tabular(question) and is_scalar_:
        return (
            "The question asks to list or show rows, but `result` is a single "
            f"value ({repr(result)}). Return a DataFrame or list instead."
        )

    return None  # plausible


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class OutputValidator:
    """
    Runs a structured checklist against an ExecutionResult.

    Usage:
        validator = OutputValidator()
        outcome = validator.validate(execution_result, question)
        if outcome.should_retry:
            # append outcome.retry_instruction to prompt and loop back to plan
    """

    def validate(self, execution_result: ExecutionResult, question: str) -> ValidationOutcome:
        """
        Run all checklist items in order. Returns on the first failure —
        there's no value in reporting multiple issues when we're about to retry.
        """

        # ------------------------------------------------------------------
        # 1. Did the executor itself report a failure?
        #    Promote executor-level retry instructions directly into the outcome
        #    rather than duplicating the messaging.
        # ------------------------------------------------------------------

        status = execution_result.status

        if status == ExecutionStatus.FORBIDDEN_IMPORT:
            return ValidationOutcome(
                passed=False,
                failure_type=ValidationFailureType.FORBIDDEN_IMPORT,
                user_message=execution_result.error_message,
                retry_instruction=execution_result.retry_instruction,
            )

        if status == ExecutionStatus.TIMEOUT:
            return ValidationOutcome(
                passed=False,
                failure_type=ValidationFailureType.TIMEOUT,
                user_message=execution_result.error_message,
                retry_instruction=execution_result.retry_instruction,
            )

        if status == ExecutionStatus.MISSING_RESULT:
            return ValidationOutcome(
                passed=False,
                failure_type=ValidationFailureType.MISSING_RESULT,
                user_message=execution_result.error_message,
                retry_instruction=execution_result.retry_instruction,
            )

        if status == ExecutionStatus.RUNTIME_ERROR:
            return ValidationOutcome(
                passed=False,
                failure_type=ValidationFailureType.EXECUTION_ERROR,
                user_message=execution_result.error_message,
                retry_instruction=execution_result.retry_instruction,
            )

        # ------------------------------------------------------------------
        # 2. Is result empty?
        #    Checked before shape — an empty DataFrame has no plausible shape.
        # ------------------------------------------------------------------

        if _is_empty(execution_result.result):
            user_msg = (
                "The code ran successfully but produced an empty result. "
                "This often means a filter is too strict or a column name is wrong."
            )
            retry = (
                "Your code executed without errors but `result` is empty. "
                "Check that:\n"
                "  - Column names match the dataset schema exactly (case-sensitive)\n"
                "  - Filter conditions are not excluding all rows\n"
                "  - The aggregation has data to operate on\n"
                "Re-examine the schema above and rewrite the code."
            )
            return ValidationOutcome(
                passed=False,
                failure_type=ValidationFailureType.EMPTY_RESULT,
                user_message=user_msg,
                retry_instruction=retry,
            )

        # ------------------------------------------------------------------
        # 3. Is the shape of result plausible given the question?
        # ------------------------------------------------------------------

        mismatch = _check_shape_plausibility(execution_result.result, question)
        if mismatch:
            return ValidationOutcome(
                passed=False,
                failure_type=ValidationFailureType.IMPLAUSIBLE_SHAPE,
                user_message=(
                    "The code ran but the output type doesn't match what the "
                    "question is asking for."
                ),
                retry_instruction=mismatch,
            )

        # ------------------------------------------------------------------
        # All checks passed
        # ------------------------------------------------------------------

        return ValidationOutcome(passed=True)