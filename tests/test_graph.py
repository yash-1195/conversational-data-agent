"""
Tests for src/agent/graph.py

Three layers of coverage:

  1. Code fence stripping (TestStripCodeFences)
       - Pure function; verifies fenced and bare-fence variants are stripped,
         plain code and partial fences are returned unchanged.

  2. Routing functions (TestRouteFromPlan, TestRouteFromEvaluate)
       - Pure functions; no mocks needed.  Pass partial AgentState dicts.
       - Exhaustively cover every branch in _route_from_plan and
         _make_route_from_evaluate.

  3. Node unit tests (TestPlanNode, TestExecuteNode, TestEvaluateNode,
                      TestRespondNode)
       - Each GraphNodes method is exercised in isolation.
       - LLMClient is replaced with Mock, subprocess calls are patched.
       - Only the state fields that the node under test actually reads are
         populated; everything else is given a safe default via
         make_initial_state and then overwritten as needed.

  4. Graph integration (TestGraphIntegration)
       - build_graph().invoke() is called end-to-end with a real DataFrame
         pickle written to a temp directory.
       - LLMClient.call is mocked to return deterministic code strings.
       - Covers three paths: happy path, clarification shortcut, and
         exhausted-retry error path.
"""

from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from src.agent.clarification import ClarificationResult
from src.agent.graph import (
    GraphNodes,
    _make_route_from_evaluate,
    _route_from_plan,
    _strip_code_fences,
    build_graph,
)
from src.agent.state import AgentState, make_initial_state
from src.agent.tools.code_executor import ExecutionResult, ExecutionStatus
from src.agent.validators.output_validator import ValidationFailureType, ValidationOutcome
from src.core.llm_client import LLMClient
from src.ingestion.profiler import DataProfile


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

def _profile() -> DataProfile:
    """Minimal DataProfile with no columns — sufficient for most node tests."""
    return DataProfile(dataset_name="test.csv", row_count=3, col_count=2, columns=[])


def _config(max_plan_attempts: int = 3, timeout_s: int = 10) -> MagicMock:
    """MagicMock with the AppConfig attributes that nodes actually read."""
    cfg = MagicMock()
    cfg.agent.max_plan_attempts = max_plan_attempts
    cfg.execution.timeout_s = timeout_s
    return cfg


def _mock_client(code: str = "result = 42") -> Mock:
    """Mock LLMClient whose .call() returns the given code string."""
    client = Mock(spec=LLMClient)
    client.call.return_value = code
    return client


def _failed_outcome(
    failure_type: ValidationFailureType = ValidationFailureType.EMPTY_RESULT,
) -> ValidationOutcome:
    return ValidationOutcome(
        passed=False,
        failure_type=failure_type,
        user_message="Something went wrong.",
        retry_instruction="Fix the code.",
    )


def _passed_outcome() -> ValidationOutcome:
    return ValidationOutcome(passed=True)


# ---------------------------------------------------------------------------
# 1. Code fence stripping
# ---------------------------------------------------------------------------

class TestStripCodeFences:
    """_strip_code_fences — removes ```python/``` wrappers from LLM output."""

    def test_python_fence_stripped(self):
        fenced = "```python\nresult = df.shape[0]\n```"
        assert _strip_code_fences(fenced) == "result = df.shape[0]"

    def test_bare_fence_stripped(self):
        fenced = "```\nresult = 42\n```"
        assert _strip_code_fences(fenced) == "result = 42"

    def test_plain_code_returned_unchanged(self):
        plain = "result = df['revenue'].sum()"
        assert _strip_code_fences(plain) == plain

    def test_multiline_code_inside_fence_preserved(self):
        fenced = "```python\nimport pandas as pd\nresult = df.groupby('region').sum()\n```"
        expected = "import pandas as pd\nresult = df.groupby('region').sum()"
        assert _strip_code_fences(fenced) == expected

    def test_leading_trailing_whitespace_ignored(self):
        fenced = "  ```python\nresult = 1\n```  "
        assert _strip_code_fences(fenced) == "result = 1"

    def test_partial_fence_returned_unchanged(self):
        # Only an opening fence — should not crash, return as-is
        partial = "```python\nresult = 1"
        assert _strip_code_fences(partial) == partial


# ---------------------------------------------------------------------------
# 3. Routing functions
# ---------------------------------------------------------------------------

class TestRouteFromPlan:
    """_route_from_plan branches on state["needs_clarification"]."""

    def test_needs_clarification_routes_to_respond(self):
        state = cast(AgentState, {"needs_clarification": True})
        assert _route_from_plan(state) == "respond"

    def test_no_clarification_routes_to_execute(self):
        state = cast(AgentState, {"needs_clarification": False})
        assert _route_from_plan(state) == "execute"


class TestRouteFromEvaluate:
    """_make_route_from_evaluate(config) branches on outcome + attempt_count."""

    def _route(self, state: AgentState, max_attempts: int = 3):
        return _make_route_from_evaluate(_config(max_plan_attempts=max_attempts))(state)

    def test_none_outcome_routes_to_respond(self):
        state = cast(AgentState, {"validation_outcome": None, "attempt_count": 0})
        assert self._route(state) == "respond"

    def test_passed_outcome_routes_to_respond(self):
        state = cast(AgentState, {
            "validation_outcome": _passed_outcome(),
            "attempt_count": 0,
        })
        assert self._route(state) == "respond"

    def test_failed_outcome_within_cap_routes_to_plan(self):
        # attempt_count = 1, max = 3 → still have retries left
        state = cast(AgentState, {
            "validation_outcome": _failed_outcome(),
            "attempt_count": 1,
        })
        assert self._route(state, max_attempts=3) == "plan"

    def test_failed_outcome_at_cap_routes_to_respond(self):
        # attempt_count = 3, max = 3 → cap reached, stop retrying
        state = cast(AgentState, {
            "validation_outcome": _failed_outcome(),
            "attempt_count": 3,
        })
        assert self._route(state, max_attempts=3) == "respond"

    def test_failed_outcome_exceeds_cap_routes_to_respond(self):
        # Defensive: attempt_count > max should still stop
        state = cast(AgentState, {
            "validation_outcome": _failed_outcome(),
            "attempt_count": 5,
        })
        assert self._route(state, max_attempts=3) == "respond"


# ---------------------------------------------------------------------------
# 4. Node unit tests
# ---------------------------------------------------------------------------

class TestPlanNode:
    """GraphNodes.plan() — ambiguity check, LLM call, retry prompt."""

    def _nodes(self, code: str = "result = 42") -> GraphNodes:
        return GraphNodes(client=_mock_client(code), config=_config())

    def _state(self, **overrides) -> AgentState:
        state = make_initial_state(
            question="how many rows?",
            dataframe_pickle_path="/tmp/dummy.pkl",
            profile=_profile(),
        )
        state.update(overrides)  # type: ignore[typeddict-item]
        return state

    def test_no_clarification_calls_llm_and_returns_code(self):
        nodes = self._nodes("result = df.shape[0]")
        state = self._state()

        with patch.object(nodes._clarification_tool, "check") as mock_check:
            mock_check.return_value = ClarificationResult(needs_clarification=False)
            updates = nodes.plan(state)

        assert updates["generated_code"] == "result = df.shape[0]"
        assert updates["needs_clarification"] is False

    def test_fenced_code_from_llm_is_stripped(self):
        """LLM response wrapped in ```python ... ``` fences must be stripped
        before being stored as generated_code, or the sandbox will fail with
        a SyntaxError on the backtick line."""
        fenced = "```python\nresult = df.shape[0]\n```"
        nodes = self._nodes(fenced)
        state = self._state()

        with patch.object(nodes._clarification_tool, "check") as mock_check:
            mock_check.return_value = ClarificationResult(needs_clarification=False)
            updates = nodes.plan(state)

        assert updates["generated_code"] == "result = df.shape[0]"
        assert "`" not in updates["generated_code"]

    def test_clarification_skips_llm_and_sets_flag(self):
        nodes = self._nodes()
        state = self._state(question="show me recent data")

        with patch.object(nodes._clarification_tool, "check") as mock_check:
            mock_check.return_value = ClarificationResult(
                needs_clarification=True,
                clarifying_question="Which date range did you have in mind?",
            )
            updates = nodes.plan(state)

        assert updates["needs_clarification"] is True
        assert updates["clarifying_question"] == "Which date range did you have in mind?"
        cast(Mock, nodes._client).call.assert_not_called()

    def test_llm_called_once_per_plan_invocation(self):
        nodes = self._nodes()
        state = self._state()

        with patch.object(nodes._clarification_tool, "check") as mock_check:
            mock_check.return_value = ClarificationResult(needs_clarification=False)
            nodes.plan(state)

        cast(Mock, nodes._client).call.assert_called_once()

    def test_retry_state_is_forwarded_to_prompt_builder(self):
        """On attempt_count > 0 the prompt builder receives failure_history."""
        nodes = self._nodes()
        state = self._state(
            attempt_count=1,
            failure_history=[(0, "empty_result", "assign a non-empty result")],
        )

        with patch.object(nodes._clarification_tool, "check") as mock_check, \
             patch.object(nodes._builder, "build_system_prompt") as mock_syspt:
            mock_check.return_value = ClarificationResult(needs_clarification=False)
            mock_syspt.return_value = "system prompt"
            nodes.plan(state)

        _, kwargs = mock_syspt.call_args
        assert kwargs["attempt_count"] == 1
        assert len(kwargs["failure_history"]) == 1


class TestExecuteNode:
    """GraphNodes.execute() — delegates to execute_code with config timeout."""

    def _nodes(self) -> GraphNodes:
        return GraphNodes(client=_mock_client(), config=_config(timeout_s=15))

    def test_execute_passes_timeout_from_config(self):
        nodes = self._nodes()
        state = make_initial_state(
            question="q",
            dataframe_pickle_path="/tmp/df.pkl",
            profile=_profile(),
            conversation_history=[],
        )
        state["generated_code"] = "result = 1"

        fake_result = ExecutionResult(status=ExecutionStatus.SUCCESS, result=1)

        with patch("src.agent.graph.execute_code", return_value=fake_result) as mock_exec:
            updates = nodes.execute(state)

        _, kwargs = mock_exec.call_args
        assert kwargs["timeout_s"] == 15
        assert updates["execution_result"] is fake_result

    def test_execute_stores_failure_result_unchanged(self):
        """ensure failed ExecutionResults are passed through, not swallowed."""
        nodes = self._nodes()
        state = make_initial_state("q", "/tmp/df.pkl", _profile())
        state["generated_code"] = "result = 1/0"

        failed_result = ExecutionResult(
            status=ExecutionStatus.RUNTIME_ERROR,
            error_message="ZeroDivisionError",
        )

        with patch("src.agent.graph.execute_code", return_value=failed_result):
            updates = nodes.execute(state)

        assert updates["execution_result"].status == ExecutionStatus.RUNTIME_ERROR


class TestEvaluateNode:
    """GraphNodes.evaluate() — validates result, increments counters on failure."""

    def _nodes(self) -> GraphNodes:
        return GraphNodes(client=_mock_client(), config=_config())

    def _state_with_result(self, result: ExecutionResult) -> AgentState:
        state = make_initial_state("q", "/tmp/dummy.pkl", _profile())
        state["execution_result"] = result
        return state

    def test_passed_outcome_does_not_increment_attempt_count(self):
        nodes = self._nodes()
        success = ExecutionResult(status=ExecutionStatus.SUCCESS, result=42)
        state = self._state_with_result(success)

        with patch.object(nodes._validator, "validate", return_value=_passed_outcome()):
            updates = nodes.evaluate(state)

        assert "attempt_count" not in updates

    def test_failed_outcome_increments_attempt_count(self):
        nodes = self._nodes()
        success = ExecutionResult(status=ExecutionStatus.SUCCESS, result=None)
        state = self._state_with_result(success)
        state["attempt_count"] = 1

        with patch.object(nodes._validator, "validate", return_value=_failed_outcome()):
            updates = nodes.evaluate(state)

        assert updates["attempt_count"] == 2

    def test_failed_outcome_appends_to_failure_history(self):
        nodes = self._nodes()
        success = ExecutionResult(status=ExecutionStatus.SUCCESS, result=None)
        state = self._state_with_result(success)
        state["attempt_count"] = 0
        state["failure_history"] = []

        outcome = _failed_outcome(ValidationFailureType.EMPTY_RESULT)
        with patch.object(nodes._validator, "validate", return_value=outcome):
            updates = nodes.evaluate(state)

        history = updates["failure_history"]
        assert len(history) == 1
        attempt_idx, failure_str, instruction = history[0]
        assert attempt_idx == 0
        assert failure_str == ValidationFailureType.EMPTY_RESULT.value

    def test_none_execution_result_raises_assertion(self):
        nodes = self._nodes()
        state = make_initial_state("q", "/tmp/dummy.pkl", _profile())
        # execution_result is None by default in make_initial_state

        with pytest.raises(RuntimeError, match="no execution_result"):
            nodes.evaluate(state)


class TestRespondNode:
    """GraphNodes.respond() — three paths: clarification, error, success."""

    def _nodes(self) -> GraphNodes:
        return GraphNodes(client=_mock_client(), config=_config())

    def _state(self, **overrides) -> AgentState:
        state = make_initial_state("q", "/tmp/dummy.pkl", _profile())
        state.update(overrides)  # type: ignore[typeddict-item]
        return state

    def test_clarification_path_returns_clarifying_question(self):
        nodes = self._nodes()
        state = self._state(
            needs_clarification=True,
            clarifying_question="Which date range?",
        )
        updates = nodes.respond(state)
        assert updates["final_answer"] == "Which date range?"
        assert updates["answer_type"] == "clarification"

    def test_error_path_returns_user_message_and_error_type(self):
        nodes = self._nodes()
        state = self._state(
            needs_clarification=False,
            validation_outcome=_failed_outcome(),
            attempt_count=3,
        )
        updates = nodes.respond(state)
        assert "Something went wrong." in updates["final_answer"]
        assert updates["answer_type"] == "error"

    def test_success_path_converts_result_to_string(self):
        nodes = self._nodes()
        exec_result = ExecutionResult(status=ExecutionStatus.SUCCESS, result=42)
        state = self._state(
            needs_clarification=False,
            validation_outcome=_passed_outcome(),
            execution_result=exec_result,
        )
        updates = nodes.respond(state)
        assert updates["final_answer"] == "42"
        assert updates["answer_type"] == "text"

    def test_success_path_without_execution_result_raises(self):
        nodes = self._nodes()
        state = self._state(
            needs_clarification=False,
            validation_outcome=_passed_outcome(),
            # execution_result remains None
        )
        with pytest.raises(RuntimeError, match="execution_result is None"):
            nodes.respond(state)


# ---------------------------------------------------------------------------
# 5. Graph integration tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def df_pickle(tmp_path: Path) -> str:
    """Write a small DataFrame to a temp pickle and return its path."""
    df = pd.DataFrame({"value": [10, 20, 30], "region": ["North", "South", "East"]})
    pickle_path = tmp_path / "test_df.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(df, f)
    return str(pickle_path)


class TestGraphIntegration:
    """
    Full build_graph().invoke() runs.

    LLMClient.call is always mocked — no real API calls. The DataFrame
    pickle is real so execute_code runs a genuine sandbox subprocess.
    """

    def _build(self, code: str = "result = int(df['value'].sum())", max_attempts: int = 3):
        client = _mock_client(code)
        config = _config(max_plan_attempts=max_attempts)
        graph = build_graph(client=client, config=config)
        return graph, client

    def _initial_state(self, df_pickle: str) -> AgentState:
        return make_initial_state(
            question="what is the total value?",
            dataframe_pickle_path=df_pickle,
            profile=_profile(),
        )

    def test_happy_path_returns_final_answer(self, df_pickle: str):
        """LLM returns valid code → execution succeeds → final_answer is set."""
        graph, _ = self._build("result = int(df['value'].sum())")
        state = self._initial_state(df_pickle)

        result = graph.invoke(state)

        assert result["final_answer"] is not None
        assert result["answer_type"] == "text"
        assert result["final_answer"] == "60"  # sum of [10, 20, 30]

    def test_happy_path_attempt_count_is_zero(self, df_pickle: str):
        """A clean first-pass execution should require no retries."""
        graph, _ = self._build("result = int(df['value'].sum())")
        state = self._initial_state(df_pickle)

        result = graph.invoke(state)

        assert result["attempt_count"] == 0

    def test_clarification_path_skips_execution(self, df_pickle: str):
        """ClarificationTool fires → no code generated, answer_type=clarification."""
        graph, client = self._build()
        state = self._initial_state(df_pickle)

        with patch(
            "src.agent.graph.ClarificationTool.check",
            return_value=ClarificationResult(
                needs_clarification=True,
                clarifying_question="Which column?",
            ),
        ):
            result = graph.invoke(state)

        assert result["answer_type"] == "clarification"
        assert result["final_answer"] == "Which column?"
        client.call.assert_not_called()

    def test_exhausted_retry_produces_error_answer(self, df_pickle: str):
        """Code that always returns None → retries exhaust → error answer."""
        # max_plan_attempts=2: evaluate fires twice (attempt_count becomes 2),
        # then routing caps and sends to respond with the error path.
        graph, client = self._build(code="result = None", max_attempts=2)
        state = self._initial_state(df_pickle)

        # Suppress clarification so the plan node always attempts code generation
        with patch(
            "src.agent.graph.ClarificationTool.check",
            return_value=ClarificationResult(needs_clarification=False),
        ):
            result = graph.invoke(state)

        assert result["answer_type"] == "error"
        assert result["attempt_count"] == 2  # incremented twice by evaluate
        # LLM was called once per attempt (2 total)
        assert client.call.call_count == 2
