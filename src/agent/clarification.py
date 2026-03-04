"""
ClarificationTool — ambiguity detection that runs before code generation.

Four heuristics are checked in the `plan` node before any LLM code generation
attempt. If any fires, the agent asks the user a targeted clarifying question
instead of guessing. This is one of the most visible behaviours in a demo —
a vague question getting a sensible follow-up question signals that the agent
understands what it doesn't know.

Heuristics (in check order):
  1. Column name ambiguity   — question references a term matching multiple columns
  2. Unspecified aggregation — aggregation implied but type not specified
  3. Unbounded time range    — time reference with no clear start or end
  4. Implied grouping        — "by" or "per" used without naming the grouping column
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.ingestion.profiler import DataProfile


# ---------------------------------------------------------------------------
# Ambiguity taxonomy
# ---------------------------------------------------------------------------

class AmbiguityType(Enum):
    COLUMN_AMBIGUITY      = "column_ambiguity"
    UNSPECIFIED_AGG       = "unspecified_aggregation"
    UNBOUNDED_TIME_RANGE  = "unbounded_time_range"
    IMPLIED_GROUPING      = "implied_grouping"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClarificationResult:
    """
    Outcome of running the ambiguity heuristics against a question.

    If `needs_clarification` is True, `clarifying_question` is a
    targeted, user-facing question to surface in the chat interface.
    The `ambiguity_type` identifies which heuristic fired, for logging.
    """
    needs_clarification: bool

    ambiguity_type: Optional[AmbiguityType] = None

    # Shown to the user when needs_clarification is True
    clarifying_question: str = ""

    # Internal detail for MLflow logging
    detail: str = ""


# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

# Verbs / nouns that imply an aggregation is wanted but don't specify which
_AGG_TRIGGERS = frozenset({
    "average", "typical", "usual", "aggregate", "summarise", "summarize",
    "combine", "overall", "general", "representative",
})

# Aggregation types that ARE explicit — if one of these is present the
# unspecified-agg heuristic does NOT fire
_EXPLICIT_AGGS = frozenset({
    "mean", "median", "mode", "sum", "total", "count", "max", "maximum",
    "min", "minimum", "std", "standard deviation", "variance", "percentile",
    "quantile", "proportion", "ratio", "percent", "percentage",
})

# Words that reference time without bounding it
_TIME_REFERENCE_WORDS = frozenset({
    "recently", "lately", "soon", "early", "late", "old", "new", "latest",
    "recent", "current", "past", "previous", "last", "next", "historical",
    "today", "yesterday", "now", "trends", "over time", "across time",
    "time period", "period", "era",
})

# Words that anchor a time reference (make it bounded enough to act on).
# Deliberately excludes "from", "to" (too generic — appear in non-temporal
# contexts like "sales from the north region") and "may" (auxiliary verb).
_TIME_ANCHORS = frozenset({
    "january", "february", "march", "april", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "q1", "q2", "q3", "q4",
    "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026",
    "week", "month", "year", "day", "hour", "quarter",
    "since", "until", "before", "after", "between",
})

# Words that imply grouping without naming the dimension
_GROUPING_TRIGGERS = frozenset({
    " by ", " per ", " for each ", " across ", " broken down by ",
    " split by ", " segmented by ", " categorised by ", " categorized by ",
    " grouped by ",
})

# Column name similarity threshold — how many characters of a question term
# must overlap with a column name before it's considered a match
_MIN_COLUMN_MATCH_CHARS = 3


# ---------------------------------------------------------------------------
# Matching helper
# ---------------------------------------------------------------------------

def _word_in(word: str, text: str) -> bool:
    """Return True if `word` (or multi-word phrase) appears as a complete
    token in `text`, using regex word boundaries so that e.g. 'count' does
    not match inside 'account' or 'discount'."""
    return bool(re.search(r"\b" + re.escape(word) + r"\b", text))


# ---------------------------------------------------------------------------
# Internal heuristic functions
# ---------------------------------------------------------------------------

def _check_column_ambiguity(
    question: str,
    profile: DataProfile,
) -> Optional[ClarificationResult]:
    """
    Fire if the question contains a term that partially matches more than
    one column name, making it unclear which column is intended.

    Match strategy: a question word is considered to reference a column if
    the column name (lowercased) appears as a substring of the word or vice
    versa, and the overlap is at least _MIN_COLUMN_MATCH_CHARS characters.
    """
    q_lower = question.lower()
    column_names = [col.name.lower() for col in profile.columns]

    # For each column, check if the question references it
    matched_columns: list[str] = []
    for col_name in column_names:
        if len(col_name) >= _MIN_COLUMN_MATCH_CHARS and col_name in q_lower:
            matched_columns.append(col_name)

    # Ambiguity only if multiple columns are referenced in one question
    # AND the question doesn't already name all of them explicitly enough
    # (heuristic: if 2+ columns matched and the question is short, it's ambiguous)
    if len(matched_columns) >= 2:
        # Suppress if matched columns flank a grouping trigger —
        # e.g. "revenue by region" is unambiguous: each column has a clear role.
        for phrase in _GROUPING_TRIGGERS:
            if phrase in q_lower:
                trigger_pos = q_lower.index(phrase)
                cols_before = [c for c in matched_columns if q_lower.index(c) < trigger_pos]
                cols_after  = [c for c in matched_columns if q_lower.index(c) > trigger_pos]
                if cols_before and cols_after:
                    return None  # roles are clear; not ambiguous

        original_names = [
            col.name for col in profile.columns
            if col.name.lower() in matched_columns
        ]
        return ClarificationResult(
            needs_clarification=True,
            ambiguity_type=AmbiguityType.COLUMN_AMBIGUITY,
            clarifying_question=(
                f"Your question could refer to multiple columns: "
                f"{', '.join(repr(n) for n in original_names)}. "
                f"Which column would you like to use?"
            ),
            detail=f"Matched columns: {original_names}",
        )

    return None


def _check_unspecified_aggregation(
    question: str,
) -> Optional[ClarificationResult]:
    """
    Fire if the question implies an aggregation (e.g. "average", "typical")
    but does not specify which kind (mean vs median vs mode etc.).

    Does NOT fire if an explicit aggregation type is already present.
    """
    q_lower = question.lower()

    # If an explicit aggregation is already named, no ambiguity.
    # Use word-boundary matching so "count" inside "account" or "discount"
    # does not falsely suppress the check.
    if any(_word_in(agg, q_lower) for agg in _EXPLICIT_AGGS):
        return None

    # Check for vague aggregation triggers (word-boundary safe).
    triggered_by = next(
        (word for word in _AGG_TRIGGERS if _word_in(word, q_lower)),
        None,
    )
    if triggered_by:
        return ClarificationResult(
            needs_clarification=True,
            ambiguity_type=AmbiguityType.UNSPECIFIED_AGG,
            clarifying_question=(
                f"You mentioned '{triggered_by}' — which aggregation would you like? "
                f"For example: mean, median, or mode?"
            ),
            detail=f"Triggered by: '{triggered_by}'",
        )

    return None


def _check_unbounded_time_range(
    question: str,
    profile: DataProfile,
) -> Optional[ClarificationResult]:
    """
    Fire if the question contains a time reference word but no anchor
    that makes the range concrete, AND the dataset has a date-like column.

    Only fires when the dataset has a date column — no point asking for
    a time range if the data has no temporal dimension.
    """
    q_lower = question.lower()

    # Only relevant if the dataset has a potential date column
    has_date_column = any(
        "date" in col.name.lower()
        or "time" in col.name.lower()
        or "possible date column stored as string" in col.flags
        or col.dtype in ("datetime64[ns]", "datetime64[ns, UTC]")
        for col in profile.columns
    )
    if not has_date_column:
        return None

    # Check for vague time reference — sort longest first so "latest" matches
    # before "late", "over time" before "time", etc. Use word-boundary matching
    # so e.g. "new" doesn't fire inside "newcomer".
    triggered_by = next(
        (word for word in sorted(_TIME_REFERENCE_WORDS, key=len, reverse=True)
         if _word_in(word, q_lower)),
        None,
    )
    if not triggered_by:
        return None

    # If any time anchor is present, the range is specific enough.
    # Word-boundary matching prevents years/months inside larger tokens.
    if any(_word_in(anchor, q_lower) for anchor in _TIME_ANCHORS):
        return None

    return ClarificationResult(
        needs_clarification=True,
        ambiguity_type=AmbiguityType.UNBOUNDED_TIME_RANGE,
        clarifying_question=(
            f"You mentioned '{triggered_by}' — could you specify the time range "
            f"you have in mind? For example: a specific month, year, or date range."
        ),
        detail=f"Triggered by: '{triggered_by}', no time anchor found",
    )


def _check_implied_grouping(
    question: str,
    profile: DataProfile,
) -> Optional[ClarificationResult]:
    """
    Fire if the question implies grouping (e.g. 'sales by ...') but the
    grouping dimension is not a recognisable column name.

    Strategy: look for a grouping trigger phrase, then check whether the
    word(s) following it match a column name. If not, ask which column
    to group by.
    """
    q_lower = question.lower()
    column_names_lower = {col.name.lower() for col in profile.columns}

    triggered_by = next(
        (phrase for phrase in _GROUPING_TRIGGERS if phrase in q_lower),
        None,
    )
    if not triggered_by:
        return None

    # Extract the token(s) immediately after the trigger phrase
    trigger_pos = q_lower.index(triggered_by)
    after_trigger = q_lower[trigger_pos + len(triggered_by):].strip()

    # Take the first 1-3 words after the trigger as the candidate dimension
    candidate_words = after_trigger.split()[:3]

    # Check if any candidate word or bigram matches a column name
    candidates_to_check = candidate_words + [
        " ".join(candidate_words[:2]),
        " ".join(candidate_words[:3]),
    ]
    dimension_found = any(
        cand.strip("?.,") in column_names_lower
        for cand in candidates_to_check
        if cand.strip("?.,")
    )

    if not dimension_found:
        col_names_display = ", ".join(
            repr(col.name) for col in profile.columns
        )
        return ClarificationResult(
            needs_clarification=True,
            ambiguity_type=AmbiguityType.IMPLIED_GROUPING,
            clarifying_question=(
                f"You'd like to group the data, but it's not clear by which column. "
                f"The available columns are: {col_names_display}. "
                f"Which one would you like to group by?"
            ),
            detail=(
                f"Trigger: '{triggered_by.strip()}', "
                f"candidate dimension: '{' '.join(candidate_words)}' "
                f"not found in column names"
            ),
        )

    return None


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class ClarificationTool:
    """
    Runs four ambiguity heuristics against a question and dataset profile.

    Returns on the first heuristic that fires — there is no value in
    stacking multiple clarifying questions at once. The priority order
    is: column ambiguity → unspecified aggregation → unbounded time range
    → implied grouping. Column ambiguity is checked first because it is
    the most structurally blocking issue.

    Usage:
        tool = ClarificationTool()
        result = tool.check(question, profile)
        if result.needs_clarification:
            # surface result.clarifying_question to the user
            # do not proceed to code generation
    """

    def check(
        self,
        question: str,
        profile: DataProfile,
    ) -> ClarificationResult:
        """
        Run all heuristics in priority order. Returns on first match.
        Returns a passing result if no heuristic fires.
        """

        # 1. Column name ambiguity
        result = _check_column_ambiguity(question, profile)
        if result:
            return result

        # 2. Unspecified aggregation
        result = _check_unspecified_aggregation(question)
        if result:
            return result

        # 3. Unbounded time range
        result = _check_unbounded_time_range(question, profile)
        if result:
            return result

        # 4. Implied grouping without named dimension
        result = _check_implied_grouping(question, profile)
        if result:
            return result

        return ClarificationResult(needs_clarification=False)