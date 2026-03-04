"""
Tests for src/agent/state.py

Verifies the make_initial_state factory and AgentState field defaults.
"""

from src.agent.state import AgentState, make_initial_state


class TestMakeInitialState:

    def test_question_is_set(self):
        state = make_initial_state("how many rows?", "/tmp/df.pkl")
        assert state["question"] == "how many rows?"

    def test_dataframe_path_is_set(self):
        state = make_initial_state("how many rows?", "/tmp/df.pkl")
        assert state["dataframe_pickle_path"] == "/tmp/df.pkl"

    def test_attempt_count_starts_at_zero(self):
        state = make_initial_state("q", "/tmp/df.pkl")
        assert state["attempt_count"] == 0

    def test_failure_history_starts_empty(self):
        state = make_initial_state("q", "/tmp/df.pkl")
        assert state["failure_history"] == []

    def test_generated_code_starts_empty(self):
        state = make_initial_state("q", "/tmp/df.pkl")
        assert state["generated_code"] == ""

    def test_execution_result_starts_none(self):
        state = make_initial_state("q", "/tmp/df.pkl")
        assert state["execution_result"] is None

    def test_validation_outcome_starts_none(self):
        state = make_initial_state("q", "/tmp/df.pkl")
        assert state["validation_outcome"] is None

    def test_needs_clarification_starts_false(self):
        state = make_initial_state("q", "/tmp/df.pkl")
        assert state["needs_clarification"] is False

    def test_clarifying_question_starts_empty(self):
        state = make_initial_state("q", "/tmp/df.pkl")
        assert state["clarifying_question"] == ""

    def test_final_answer_starts_none(self):
        state = make_initial_state("q", "/tmp/df.pkl")
        assert state["final_answer"] is None

    def test_answer_type_starts_empty(self):
        state = make_initial_state("q", "/tmp/df.pkl")
        assert state["answer_type"] == ""

    def test_conversation_history_defaults_to_empty_list(self):
        state = make_initial_state("q", "/tmp/df.pkl")
        assert state["conversation_history"] == []

    def test_conversation_history_can_be_provided(self):
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        state = make_initial_state("q", "/tmp/df.pkl", conversation_history=history)
        assert state["conversation_history"] == history

    def test_none_conversation_history_becomes_empty_list(self):
        state = make_initial_state("q", "/tmp/df.pkl", conversation_history=None)
        assert state["conversation_history"] == []