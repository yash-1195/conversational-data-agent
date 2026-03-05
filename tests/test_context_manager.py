# tests/test_context_manager.py

"""
Tests for src/agent/context_manager.py

Covers:
  - Single turn appended to empty history
  - Multiple turns within the window (all kept verbatim)
  - Turns exceeding the window (older ones compressed, none dropped)
  - Compression format and truncation
  - Idempotency: re-compressing an already-compressed message is a no-op
  - Output is always even-length
  - Graceful handling of None / missing fields on completed_state
  - answer_type prefix appears correctly in verbatim assistant messages
  - Input history is never mutated

Note on window semantics
------------------------
The context manager keeps ALL turns — none are dropped. Recent turns
(within max_context_turns) are kept verbatim; older turns are compressed
to a single summary line. Total message count = 2 * total_turns always.
The window controls verbatim vs compressed, not what is retained.
"""

from typing import cast

from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessageParam

from src.agent.context_manager import (
    ContextManager,
    _compress_assistant_message,
    _compress_pair,
    _to_flat,
    _to_pairs,
)
from src.agent.state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(
    question: str,
    final_answer: str,
    answer_type: str = "text",
    conversation_history: list | None = None,
) -> AgentState:
    """Build a minimal completed AgentState dict for testing.

    Only the four fields read by ContextManager.update() are populated.
    cast() tells the type checker this satisfies AgentState; at runtime
    update() uses .get() with defaults so missing fields are safe.
    """
    return cast(AgentState, {
        "question": question,
        "final_answer": final_answer,
        "answer_type": answer_type,
        "conversation_history": conversation_history or [],
    })


def _user(content: str) -> ChatCompletionMessageParam:
    return cast(ChatCompletionMessageParam, {"role": "user", "content": content})


def _assistant(content: str) -> ChatCompletionMessageParam:
    return cast(ChatCompletionMessageParam, {"role": "assistant", "content": content})


def _get_content(msg: ChatCompletionMessageParam) -> str:
    """Extract string content from a message param for test assertions."""
    return cast(str, msg.get("content") or "")  # type: ignore[union-attr]


def _build_history(cm: ContextManager, num_turns: int) -> list[ChatCompletionMessageParam]:
    """Run cm.update() for num_turns sequential turns and return final history."""
    history: list[ChatCompletionMessageParam] = []
    for i in range(num_turns):
        state = _make_state(
            question=f"question {i}",
            final_answer=f"answer {i}\nextra detail {i}",
            answer_type="text",
            conversation_history=history,
        )
        history = cm.update(state)
    return history


# ---------------------------------------------------------------------------
# _to_pairs / _to_flat round-trip
# ---------------------------------------------------------------------------

class TestToPairsAndToFlat:

    def test_even_list_pairs_correctly(self):
        flat = [_user("q1"), _assistant("a1"), _user("q2"), _assistant("a2")]
        pairs = _to_pairs(flat)
        assert len(pairs) == 2
        assert pairs[0] == (_user("q1"), _assistant("a1"))
        assert pairs[1] == (_user("q2"), _assistant("a2"))

    def test_round_trip_is_identity(self):
        flat = [_user("q1"), _assistant("a1"), _user("q2"), _assistant("a2")]
        assert _to_flat(_to_pairs(flat)) == flat

    def test_odd_list_drops_trailing_message(self):
        flat = [_user("q1"), _assistant("a1"), _user("orphan")]
        pairs = _to_pairs(flat)
        assert len(pairs) == 1

    def test_empty_list_returns_empty(self):
        assert _to_pairs([]) == []
        assert _to_flat([]) == []

    def test_wrong_role_order_is_paired_positionally(self):
        # _to_pairs is positional and does not validate role order.
        # This test documents the current behaviour: a [user, user, assistant, assistant]
        # sequence is paired as (user, user) + (assistant, assistant) without error.
        # If role validation is ever added, this test should be updated to expect a
        # ValueError or similar.
        flat = [_user("q1"), _user("q2"), _assistant("a1"), _assistant("a2")]
        pairs = _to_pairs(flat)
        assert len(pairs) == 2
        assert pairs[0] == (_user("q1"), _user("q2"))


# ---------------------------------------------------------------------------
# _compress_assistant_message
# ---------------------------------------------------------------------------

class TestCompressAssistantMessage:

    def test_verbatim_message_is_compressed_to_one_line(self):
        msg = _assistant("[table]\nrevenue  region\n1000  North\n2000  South")
        result = _compress_assistant_message(cast(ChatCompletionAssistantMessageParam, msg))
        assert "\n" not in _get_content(result)

    def test_type_prefix_preserved_after_compression(self):
        msg = _assistant("[table]\nrevenue  region\n1000  North")
        result = _compress_assistant_message(cast(ChatCompletionAssistantMessageParam, msg))
        assert _get_content(result).startswith("[table]")

    def test_first_content_line_used(self):
        msg = _assistant("[text]\nThe total revenue is 42000.\nExtra line.")
        result = _compress_assistant_message(cast(ChatCompletionAssistantMessageParam, msg))
        assert "The total revenue is 42000." in _get_content(result)
        assert "Extra line." not in _get_content(result)

    def test_long_first_line_is_truncated(self):
        long_line = "x" * 300
        msg = _assistant(f"[text]\n{long_line}")
        result = _compress_assistant_message(cast(ChatCompletionAssistantMessageParam, msg))
        # Max = len("[text] ") + 200 chars + len("...") = 210
        assert len(_get_content(result)) <= 210
        assert _get_content(result).endswith("...")

    def test_already_compressed_message_is_unchanged(self):
        already = _assistant("[text] Short summary line.")
        result = _compress_assistant_message(cast(ChatCompletionAssistantMessageParam, already))
        assert result == cast(ChatCompletionAssistantMessageParam, already)

    def test_role_preserved(self):
        msg = _assistant("[plot]\nFigure description")
        result = _compress_assistant_message(cast(ChatCompletionAssistantMessageParam, msg))
        assert result["role"] == "assistant"

    def test_empty_content_returned_unchanged(self):
        msg = _assistant("")
        result = _compress_assistant_message(cast(ChatCompletionAssistantMessageParam, msg))
        assert result == cast(ChatCompletionAssistantMessageParam, msg)


# ---------------------------------------------------------------------------
# ContextManager.update — first turn
# ---------------------------------------------------------------------------

class TestContextManagerFirstTurn:

    def test_returns_two_messages(self):
        cm = ContextManager(max_context_turns=5)
        state = _make_state("how many rows?", "500 rows")
        history = cm.update(state)
        assert len(history) == 2

    def test_user_message_is_first(self):
        cm = ContextManager(max_context_turns=5)
        state = _make_state("how many rows?", "500 rows")
        history = cm.update(state)
        assert history[0]["role"] == "user"
        assert _get_content(history[0]) == "how many rows?"

    def test_assistant_message_is_second(self):
        cm = ContextManager(max_context_turns=5)
        state = _make_state("how many rows?", "500 rows", answer_type="text")
        history = cm.update(state)
        assert history[1]["role"] == "assistant"

    def test_answer_type_prefix_in_assistant_content(self):
        cm = ContextManager(max_context_turns=5)
        state = _make_state("show revenue table", "col1  col2", answer_type="table")
        history = cm.update(state)
        assert _get_content(history[1]).startswith("[table]")

    def test_full_answer_present_in_verbatim_message(self):
        cm = ContextManager(max_context_turns=5)
        answer = "The total is 42000.\nBreakdown: North 20000, South 22000."
        state = _make_state("what is total revenue?", answer, answer_type="text")
        history = cm.update(state)
        assert "The total is 42000." in _get_content(history[1])
        assert "Breakdown: North 20000, South 22000." in _get_content(history[1])


# ---------------------------------------------------------------------------
# ContextManager.update — multiple turns within window
# ---------------------------------------------------------------------------

class TestContextManagerWithinWindow:

    def test_all_turns_kept_verbatim_when_within_window(self):
        cm = ContextManager(max_context_turns=5)
        history = _build_history(cm, num_turns=3)
        # 3 turns × 2 messages = 6 messages, all verbatim
        assert len(history) == 6
        assistant_messages = [m for m in history if m["role"] == "assistant"]
        for msg in assistant_messages:
            assert "\n" in _get_content(msg)  # verbatim = multiline

    def test_output_is_always_even_length(self):
        cm = ContextManager(max_context_turns=5)
        history: list[ChatCompletionMessageParam] = []
        for i in range(4):
            state = _make_state(
                question=f"q{i}", final_answer=f"a{i}", conversation_history=history
            )
            history = cm.update(state)
            assert len(history) % 2 == 0

    def test_roles_alternate_user_assistant(self):
        cm = ContextManager(max_context_turns=5)
        history = _build_history(cm, num_turns=3)
        roles = [msg["role"] for msg in history]
        for i, role in enumerate(roles):
            expected = "user" if i % 2 == 0 else "assistant"
            assert role == expected

    def test_questions_preserved_verbatim(self):
        cm = ContextManager(max_context_turns=5)
        history: list[ChatCompletionMessageParam] = []
        questions = ["how many rows?", "what is total revenue?", "show by region"]
        for q in questions:
            state = _make_state(
                question=q, final_answer="some answer", conversation_history=history
            )
            history = cm.update(state)
        user_messages = [_get_content(m) for m in history if m["role"] == "user"]
        assert user_messages == questions


# ---------------------------------------------------------------------------
# ContextManager.update — window overflow and compression
# ---------------------------------------------------------------------------

class TestContextManagerWindowOverflow:

    def test_total_message_count_equals_two_times_turns(self):
        # No turns are dropped — all are kept, older ones compressed.
        # Total messages = 2 * num_turns always.
        cm = ContextManager(max_context_turns=3)
        history = _build_history(cm, num_turns=6)
        assert len(history) == 12  # 6 turns × 2 messages

    def test_first_assistant_is_compressed_after_overflow(self):
        cm = ContextManager(max_context_turns=2)
        history = _build_history(cm, num_turns=4)
        # Oldest assistant message is at index 1 — must be single-line
        assert "\n" not in _get_content(history[1])

    def test_last_assistant_is_verbatim_after_overflow(self):
        cm = ContextManager(max_context_turns=2)
        history = _build_history(cm, num_turns=4)
        # Most recent assistant is always last — must be multiline (verbatim)
        assert "\n" in _get_content(history[-1])

    def test_type_prefix_preserved_in_compressed_turns(self):
        cm = ContextManager(max_context_turns=1)
        history = _build_history(cm, num_turns=3)
        # Oldest assistant (index 1) is compressed but type prefix must remain
        compressed_content = _get_content(history[1])
        assert compressed_content.startswith("[text]")

    def test_window_size_one_compresses_all_but_latest(self):
        cm = ContextManager(max_context_turns=1)
        history = _build_history(cm, num_turns=5)
        # All 5 turns kept (4 compressed + 1 verbatim) = 10 messages
        assert len(history) == 10
        assistant_messages = [m for m in history if m["role"] == "assistant"]
        for msg in assistant_messages[:-1]:
            assert "\n" not in _get_content(msg)
        assert "\n" in _get_content(assistant_messages[-1])

    def test_compression_is_idempotent_across_overflow_cycles(self):
        """A turn compressed in one cycle must not be double-compressed."""
        cm = ContextManager(max_context_turns=1)
        history: list[ChatCompletionMessageParam] = []
        for i in range(5):
            state = _make_state(
                question=f"q{i}",
                final_answer=f"answer {i}\nextra",
                answer_type="text",
                conversation_history=history,
            )
            history = cm.update(state)

        assistant_messages = [m for m in history if m["role"] == "assistant"]
        for msg in assistant_messages[:-1]:
            assert "\n" not in _get_content(msg)
            # Must not have double prefix like "[text] [text] ..."
            assert _get_content(msg).count("[text]") == 1


# ---------------------------------------------------------------------------
# ContextManager.update — edge cases
# ---------------------------------------------------------------------------

class TestContextManagerEdgeCases:

    def test_none_final_answer_does_not_crash(self):
        cm = ContextManager(max_context_turns=5)
        state = cast(AgentState, {
            "question": "q",
            "final_answer": None,
            "answer_type": "text",
            "conversation_history": [],
        })
        history = cm.update(state)
        assert len(history) == 2

    def test_none_conversation_history_treated_as_empty(self):
        cm = ContextManager(max_context_turns=5)
        state = cast(AgentState, {
            "question": "q",
            "final_answer": "a",
            "answer_type": "text",
            "conversation_history": None,
        })
        history = cm.update(state)
        assert len(history) == 2

    def test_missing_answer_type_defaults_to_text(self):
        cm = ContextManager(max_context_turns=5)
        state = cast(AgentState, {"question": "q", "final_answer": "a", "conversation_history": []})
        history = cm.update(state)
        assert _get_content(history[1]).startswith("[text]")

    def test_clarification_answer_type_stored_correctly(self):
        cm = ContextManager(max_context_turns=5)
        state = _make_state(
            "show me recent data",
            "Could you specify a date range?",
            answer_type="clarification",
        )
        history = cm.update(state)
        assert _get_content(history[1]).startswith("[clarification]")

    def test_input_history_not_mutated(self):
        cm = ContextManager(max_context_turns=5)
        original_history = [_user("old q"), _assistant("[text]\nold a")]
        state = _make_state("new q", "new a", conversation_history=original_history)
        cm.update(state)
        # List not extended
        assert len(original_history) == 2
        # Dict objects inside not mutated
        assert _get_content(original_history[0]) == "old q"
        assert _get_content(original_history[1]) == "[text]\nold a"