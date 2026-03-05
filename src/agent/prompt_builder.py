"""
PromptBuilder — assembles the LLM system prompt fresh on every agent turn.

The system prompt is never cached or reused across turns. It is rebuilt
from scratch each time so the LLM always has:
  - The current dataset schema and quality flags
  - The current retry context (failure reason from previous attempt)
  - The full conversation history window

This is intentional: relying on the prompt persisting in conversation
history causes context drift and stale schema references. Regenerating
it fresh is the correct approach.

The prompt has seven sections assembled in order:
  1. Role and task definition
  2. Dataset schema (from DataProfile.to_prompt_str())
  3. Sandbox rules (security guardrails)
  4. Code formatting instructions
  5. Ambiguity handling instructions
  6. Data quality caveat instructions
  7. Few-shot examples
  + Retry context section (appended only on attempt > 0)
"""

from __future__ import annotations

from typing import Optional

from openai.types.chat import ChatCompletionMessageParam

from src.agent.tools.code_executor import ALLOWED_MODULES
from src.ingestion.profiler import DataProfile


# ---------------------------------------------------------------------------
# Static prompt sections
# ---------------------------------------------------------------------------

_ROLE_SECTION = """You are a data analysis assistant. Your job is to answer \
questions about a dataset by writing and executing Python/pandas code.

Rules you must always follow:
- Always assign your final output to a variable named exactly `result`.
- `result` can be a scalar, a pandas DataFrame, a list, a dict, or a \
matplotlib Figure — whatever best answers the question.
- If the question is ambiguous or cannot be answered from the available \
columns, say so clearly instead of guessing.
- Never fabricate data. If the answer is not in the dataset, say so."""

def _build_sandbox_rules_section() -> str:
    """Build the sandbox rules section dynamically from ALLOWED_MODULES.

    Driven from the single source of truth in code_executor.py so that
    adding a module there automatically updates the LLM's instructions.
    """
    allowed_str = ", ".join(sorted(ALLOWED_MODULES))
    return (
        "## Sandbox security rules\n\n"
        "The code you write runs in a restricted sandbox. You must follow these "
        "rules exactly or your code will be rejected before execution:\n\n"
        f"ALLOWED imports only:\n  {allowed_str}\n\n"
        "FORBIDDEN (will be rejected immediately):\n"
        "  - Any import not in the allowed list above\n"
        "  - os, sys, subprocess, socket, shutil, ctypes, threading, multiprocessing\n"
        "  - eval(), exec(), compile(), __import__()\n"
        "  - Reading or writing files (other than via the pre-loaded `df` variable)\n"
        "  - Network calls of any kind\n\n"
        "The DataFrame is already loaded as `df`. Do not re-load it."
    )

_CODE_FORMAT_SECTION = """## Code format

- Write plain Python. Do not wrap code in markdown fences or add explanations.
- The final output MUST be assigned to `result`. This is checked automatically.
- For plots: create a matplotlib Figure and assign it to `result`.
  Example: `fig, ax = plt.subplots(); ax.bar(...); result = fig`
- For tables: assign a pandas DataFrame to `result`.
- For single values: assign the scalar directly to `result`.
  Example: `result = df['revenue'].sum()`
- Keep code concise. Prefer pandas built-ins over Python loops."""

_AMBIGUITY_SECTION = """## Handling ambiguous questions

If the question has more than one reasonable interpretation, DO NOT guess. \
Instead, assign a clarifying question to `result` as a plain string.
Example: `result = "Did you mean revenue or profit?"`

Specifically, ask for clarification if:
- A column name in the question matches multiple columns in the schema
- An aggregation is implied but the type is not specified (mean vs median?)
- A time range is referenced without a concrete start or end point
- A grouping dimension is implied but not named"""

_DATA_QUALITY_SECTION = """## Data quality caveats

The dataset schema below includes quality flags for columns with known issues \
(high null rates, mixed types, possible date strings). If the column central \
to your answer has a quality flag, append a one-line caveat to the response \
text noting the potential issue.

Example caveat: "Note: the 'revenue' column has a 35% null rate — \
this result excludes those rows." """

# Few-shot examples — one good question, one ambiguous question
_FEW_SHOT_SECTION = """## Examples

Good question (clear, actionable):
  Q: "What is the total revenue by region?"
  A:
    result = df.groupby('region')['revenue'].sum().reset_index()

Good question (scalar answer):
  Q: "How many rows have a null value in the 'score' column?"
  A:
    result = df['score'].isna().sum()

Ambiguous question (ask for clarification):
  Q: "What is the average order?"
  A:
    result = "Could you clarify which column you'd like to average, \
and whether you want the mean or median?"

Ambiguous question (time range not specified):
  Q: "Show me recent sales."
  A:
    result = "Could you specify the time range? \
For example: last 30 days, Q3 2024, or a specific date range." """


# ---------------------------------------------------------------------------
# Retry context builder
# ---------------------------------------------------------------------------

def _build_retry_section(
    attempt_count: int,
    failure_history: list[tuple[int, str, str]],
) -> str:
    """
    Build the retry context section appended to the prompt on attempt > 0.

    Shows the most recent failure reason and the full failure history
    so the LLM understands the pattern of what has been tried.
    """
    # Guard on failure_history alone — if it is empty there is nothing to
    # show regardless of attempt_count, and indexing into it would crash.
    if not failure_history:
        return ""

    lines = [
        f"\n## Retry context (attempt {attempt_count + 1})",
        f"Your previous {attempt_count} attempt(s) failed. "
        "Read the failure reasons carefully and fix the specific issue.\n",
    ]

    # Most recent failure — show in full
    latest_idx, latest_type, latest_instruction = failure_history[-1]
    lines.append(f"Most recent failure (attempt {latest_idx + 1}):")
    lines.append(f"  Type: {latest_type}")
    lines.append(f"  Fix required: {latest_instruction}")

    # Earlier failures — summarise
    if len(failure_history) > 1:
        lines.append("\nPrevious failures (summary):")
        for idx, ftype, _ in failure_history[:-1]:
            lines.append(f"  Attempt {idx + 1}: {ftype}")

    lines.append(
        "\nDo NOT repeat the same approach. Address the specific fix above."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Assembles the system prompt and conversation messages for each LLM call.

    Usage:
        builder = PromptBuilder()

        # System prompt (passed as system= in the LLM call)
        system = builder.build_system_prompt(
            profile=profile,
            attempt_count=state["attempt_count"],
            failure_history=state["failure_history"],
        )

        # Messages list (passed as messages= in the LLM call)
        messages = builder.build_messages(
            question=state["question"],
            conversation_history=state["conversation_history"],
        )
    """

    def build_system_prompt(
        self,
        profile: DataProfile,
        attempt_count: int = 0,
        failure_history: Optional[list[tuple[int, str, str]]] = None,
    ) -> str:
        """
        Build the full system prompt for a single LLM call.

        Parameters
        ----------
        profile:
            The DataProfile for the loaded dataset. Regenerated fresh
            each turn — never pulled from conversation history.
        attempt_count:
            Number of attempts completed so far. 0 on the first attempt.
        failure_history:
            List of (attempt_index, failure_type_str, retry_instruction)
            tuples. Empty on the first attempt.

        Returns
        -------
        str
            The complete system prompt ready to pass to the LLM.
        """
        failure_history = failure_history or []

        sections = [
            _ROLE_SECTION,
            "## Dataset schema\n\n" + profile.to_prompt_str(),
            _build_sandbox_rules_section(),
            _CODE_FORMAT_SECTION,
            _AMBIGUITY_SECTION,
            _DATA_QUALITY_SECTION,
            _FEW_SHOT_SECTION,
        ]

        retry_section = _build_retry_section(attempt_count, failure_history)
        if retry_section:
            sections.append(retry_section)

        return "\n\n".join(sections)

    def build_messages(
        self,
        question: str,
        conversation_history: Optional[list[ChatCompletionMessageParam]] = None,
    ) -> list[ChatCompletionMessageParam]:
        """
        Build the messages list for the LLM call.

        Conversation history is prepended before the current question
        so the model has conversational context. The history is
        already compressed and windowed by context_manager — this
        method does not truncate it further.

        Parameters
        ----------
        question:
            The current user question.
        conversation_history:
            List of {"role": ..., "content": ...} dicts from
            context_manager. May be empty on the first turn.

        Returns
        -------
        list[ChatCompletionMessageParam]
            Messages list ready to pass to the LLM.
        """
        conversation_history = conversation_history or []

        messages: list[ChatCompletionMessageParam] = list(conversation_history)
        messages.append({"role": "user", "content": question})

        return messages