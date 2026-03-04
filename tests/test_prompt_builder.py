"""
Tests for src/agent/prompt_builder.py

Verifies section presence, retry context assembly, and message construction.
Tests check for key content rather than exact wording — prompt wording
changes often and exact-match tests would be brittle.
"""

from src.agent.prompt_builder import PromptBuilder
from src.ingestion.profiler import ColumnProfile, DataProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_profile(column_names: list[str]) -> DataProfile:
    columns = [
        ColumnProfile(
            name=name,
            dtype="object",
            null_count=0,
            null_rate=0.0,
            unique_count=10,
            sample_values=["a", "b"],
            flags=[],
        )
        for name in column_names
    ]
    return DataProfile(
        dataset_name="sales.csv",
        row_count=500,
        col_count=len(columns),
        columns=columns,
    )


def _make_flagged_profile() -> DataProfile:
    """Profile with a high-null column for data quality caveat tests."""
    col = ColumnProfile(
        name="revenue",
        dtype="float64",
        null_count=200,
        null_rate=0.4,
        unique_count=300,
        sample_values=[100.0, 200.0],
        flags=["high null rate (40%)"],
    )
    return DataProfile(
        dataset_name="sales.csv",
        row_count=500,
        col_count=1,
        columns=[col],
    )


BUILDER = PromptBuilder()


# ---------------------------------------------------------------------------
# build_system_prompt — section presence
# ---------------------------------------------------------------------------

class TestSystemPromptSections:

    def test_contains_dataset_schema(self):
        profile = _make_profile(["revenue", "region"])
        prompt = BUILDER.build_system_prompt(profile)
        assert "revenue" in prompt
        assert "region" in prompt
        assert "sales.csv" in prompt

    def test_contains_sandbox_rules(self):
        profile = _make_profile(["revenue"])
        prompt = BUILDER.build_system_prompt(profile)
        assert "os" in prompt
        assert "subprocess" in prompt
        assert "eval" in prompt

    def test_contains_allowed_imports(self):
        profile = _make_profile(["revenue"])
        prompt = BUILDER.build_system_prompt(profile)
        assert "pandas" in prompt
        assert "numpy" in prompt
        assert "matplotlib" in prompt

    def test_contains_result_assignment_instruction(self):
        profile = _make_profile(["revenue"])
        prompt = BUILDER.build_system_prompt(profile)
        assert "result" in prompt

    def test_contains_ambiguity_instruction(self):
        profile = _make_profile(["revenue"])
        prompt = BUILDER.build_system_prompt(profile)
        assert "clarif" in prompt.lower()

    def test_contains_few_shot_examples(self):
        profile = _make_profile(["revenue", "region"])
        prompt = BUILDER.build_system_prompt(profile)
        assert "groupby" in prompt
        assert "isna" in prompt

    def test_contains_data_quality_caveat_instruction(self):
        profile = _make_profile(["revenue"])
        prompt = BUILDER.build_system_prompt(profile)
        assert any(
            word in prompt.lower()
            for word in ["quality", "caveat", "null"]
        )

    def test_flagged_column_appears_in_schema_section(self):
        profile = _make_flagged_profile()
        prompt = BUILDER.build_system_prompt(profile)
        assert "high null rate" in prompt

    def test_schema_is_regenerated_each_call(self):
        """Two calls with different profiles must produce different prompts."""
        # Use column names unlikely to appear in static prompt sections.
        # Assert against the pipe-delimited schema format from to_prompt_str()
        # so we're specifically testing the schema section, not static examples.
        profile_a = _make_profile(["profit_margin_pct"])
        profile_b = _make_profile(["ambient_temperature"])
        prompt_a = BUILDER.build_system_prompt(profile_a)
        prompt_b = BUILDER.build_system_prompt(profile_b)
        # Schema lines look like: "  profit_margin_pct | object | nulls: ..."
        assert "profit_margin_pct | object" in prompt_a
        assert "profit_margin_pct | object" not in prompt_b
        assert "ambient_temperature | object" in prompt_b
        assert "ambient_temperature | object" not in prompt_a


# ---------------------------------------------------------------------------
# build_system_prompt — retry context
# ---------------------------------------------------------------------------

class TestRetryContext:

    def test_no_retry_header_on_first_attempt(self):
        profile = _make_profile(["revenue"])
        prompt = BUILDER.build_system_prompt(profile, attempt_count=0)
        assert "retry context" not in prompt.lower()

    def test_retry_section_present_on_second_attempt(self):
        profile = _make_profile(["revenue"])
        history = [(0, "missing_result", "assign result to `result`")]
        prompt = BUILDER.build_system_prompt(
            profile, attempt_count=1, failure_history=history
        )
        assert "attempt" in prompt.lower()
        assert "missing_result" in prompt

    def test_retry_instruction_included_in_prompt(self):
        profile = _make_profile(["revenue"])
        instruction = "Your code never assigned result — fix this specific issue"
        history = [(0, "missing_result", instruction)]
        prompt = BUILDER.build_system_prompt(
            profile, attempt_count=1, failure_history=history
        )
        assert instruction in prompt

    def test_multiple_failures_all_appear(self):
        profile = _make_profile(["revenue"])
        history = [
            (0, "missing_result", "assign to result"),
            (1, "runtime_error", "column name is wrong"),
        ]
        prompt = BUILDER.build_system_prompt(
            profile, attempt_count=2, failure_history=history
        )
        assert "missing_result" in prompt
        assert "runtime_error" in prompt

    def test_latest_failure_instruction_is_prominent(self):
        """Most recent failure instruction must appear in full in the prompt."""
        profile = _make_profile(["revenue"])
        latest_instruction = "Use df['revenue'].sum() not df.revenue.sum()"
        history = [
            (0, "runtime_error", "some earlier error"),
            (1, "runtime_error", latest_instruction),
        ]
        prompt = BUILDER.build_system_prompt(
            profile, attempt_count=2, failure_history=history
        )
        assert latest_instruction in prompt

    def test_retry_prompt_longer_than_base_prompt(self):
        profile = _make_profile(["revenue"])
        base = BUILDER.build_system_prompt(profile, attempt_count=0, failure_history=[])
        with_retry = BUILDER.build_system_prompt(
            profile, attempt_count=1,
            failure_history=[(0, "timeout", "use vectorised ops")]
        )
        assert len(with_retry) > len(base)


# ---------------------------------------------------------------------------
# build_messages
# ---------------------------------------------------------------------------

class TestBuildMessages:

    def test_question_is_last_message(self):
        messages = BUILDER.build_messages("how many rows?")
        assert messages[-1] == {"role": "user", "content": "how many rows?"}

    def test_empty_history_produces_single_message(self):
        messages = BUILDER.build_messages("how many rows?", conversation_history=[])
        assert len(messages) == 1

    def test_history_prepended_before_question(self):
        history = [
            {"role": "user", "content": "what columns exist?"},
            {"role": "assistant", "content": "revenue, region, date"},
        ]
        messages = BUILDER.build_messages("what is total revenue?", conversation_history=history)
        assert len(messages) == 3
        assert messages[0]["content"] == "what columns exist?"
        assert messages[1]["content"] == "revenue, region, date"
        assert messages[2]["content"] == "what is total revenue?"

    def test_none_history_treated_as_empty(self):
        messages = BUILDER.build_messages("q", conversation_history=None)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_history_not_mutated(self):
        """build_messages must not modify the original history list."""
        history = [{"role": "user", "content": "old question"}]
        original_len = len(history)
        BUILDER.build_messages("new question", conversation_history=history)
        assert len(history) == original_len

    def test_roles_preserved_from_history(self):
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        messages = BUILDER.build_messages("q2", conversation_history=history)
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"