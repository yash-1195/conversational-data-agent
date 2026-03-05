"""
ContextManager — maintains a sliding window of conversation history
across agent turns.

After each completed turn, the caller passes the finished AgentState to
update(). The manager appends the new turn, keeps the most recent
max_context_turns pairs verbatim, and compresses any older pairs to a
single summary line. The result is a flat list of
{"role": ..., "content": ...} dicts ready for PromptBuilder.build_messages().

Turn definition
---------------
One turn = one user question + one assistant answer = one pair of messages.
The history list is always even-length. build_messages() appends the
current (in-progress) question at the end, making it odd — correct for
an in-progress turn.

Storage format
--------------
Assistant messages are stored with a type prefix so the answer type
survives the verbatim stage and is still available when that turn is
later compressed:

  Verbatim:    "[{answer_type}]\\n{full answer text}"
  Compressed:  "[{answer_type}] {first line of answer, max 200 chars}"

This prefix is the reason answer_type is threaded through AgentState —
without it the compressor would have to infer the type from answer text.
"""

from __future__ import annotations

from typing import cast

from openai.types.chat import ChatCompletionAssistantMessageParam, ChatCompletionMessageParam

from src.agent.state import AgentState


_COMPRESSED_CONTENT_MAX_CHARS = 200


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_pairs(
    flat: list[ChatCompletionMessageParam],
) -> list[tuple[ChatCompletionMessageParam, ChatCompletionMessageParam]]:
    """
    Split a flat message list into (user, assistant) pairs.

    If the list is somehow odd-length (caller bug), the trailing
    unpaired message is silently dropped rather than crashing.

    Note: pairing is purely positional — correct role alternation
    (user at even indices, assistant at odd) is assumed and not
    validated. Callers must ensure history is well-formed before
    passing it in.
    """
    n = len(flat) - (len(flat) % 2)  # round down to even
    return [(flat[i], flat[i + 1]) for i in range(0, n, 2)]


def _to_flat(
    pairs: list[tuple[ChatCompletionMessageParam, ChatCompletionMessageParam]],
) -> list[ChatCompletionMessageParam]:
    """Flatten (user, assistant) pairs back into a single message list."""
    return [msg for pair in pairs for msg in pair]


def _compress_assistant_message(
    msg: ChatCompletionAssistantMessageParam,
) -> ChatCompletionAssistantMessageParam:
    """
    Compress a verbatim assistant message to a single summary line.

    Verbatim content format:    "[{type}]\n{full answer}"
    Compressed content format:  "[{type}] {first line, max 200 chars}"

    If the message is already compressed (single non-empty line), it is
    returned unchanged — re-compressing a compressed message is a no-op.
    """
    content = cast(str, msg.get("content") or "")  # type: ignore[union-attr]
    lines = [line for line in content.split("\n") if line.strip()]

    if len(lines) <= 1:
        # Already compressed or empty — nothing to do
        return msg

    type_prefix = lines[0]  # e.g. "[table]"

    # Guard: only compress messages that follow the expected "[type]\n..." format.
    # Messages without a recognised prefix are returned unchanged to avoid
    # silently mangling manually constructed or externally sourced history.
    if not (type_prefix.startswith("[") and type_prefix.endswith("]")):
        return msg

    first_content = lines[1].strip()

    if len(first_content) > _COMPRESSED_CONTENT_MAX_CHARS:
        first_content = first_content[:_COMPRESSED_CONTENT_MAX_CHARS] + "..."

    return {"role": "assistant", "content": f"{type_prefix} {first_content}"}


def _compress_pair(
    pair: tuple[ChatCompletionMessageParam, ChatCompletionMessageParam],
) -> tuple[ChatCompletionMessageParam, ChatCompletionMessageParam]:
    """Compress the assistant half of a (user, assistant) pair."""
    user_msg, assistant_msg = pair
    return (user_msg, _compress_assistant_message(cast(ChatCompletionAssistantMessageParam, assistant_msg)))


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class ContextManager:
    """
    Sliding window context manager for multi-turn conversation history.

    Keeps the most recent max_context_turns turns verbatim and compresses
    older turns to single-line summaries. Operates on (user, assistant)
    pairs so the window boundary never cuts a pair in half.

    This class is stateless — it owns no history itself. All conversation
    state is passed in via completed_state and returned by update(). This
    makes it safe to use as a plain utility inside a LangGraph node.

    Parameters
    ----------
    max_context_turns:
        Number of most recent turns to keep verbatim. Source from
        config.llm.max_context_turns.

    Usage
    -----
        cm = ContextManager(max_context_turns=config.llm.max_context_turns)

        # After each completed graph run:
        new_history = cm.update(completed_state)

        # Pass into the next turn:
        state = make_initial_state(
            question=next_question,
            dataframe_pickle_path=path,
            profile=profile,
            conversation_history=new_history,
        )
    """

    def __init__(self, max_context_turns: int) -> None:
        self._max_context_turns = max_context_turns

    def update(self, completed_state: AgentState) -> list[ChatCompletionMessageParam]:
        """
        Append the completed turn to history and apply the sliding window.

        Called once per completed graph run, after the respond node has set
        final_answer and answer_type on the state. The existing
        conversation_history on the state is the already-windowed output
        from the previous call — this method extends it with the new turn
        and re-applies the window.

        Parameters
        ----------
        completed_state:
            The AgentState after graph execution is complete. Must have
            question, final_answer, answer_type, and conversation_history
            populated (i.e. the respond node has run).

        Returns
        -------
        list[ChatCompletionMessageParam]
            Flat list of OpenAI-compatible message dicts, always
            even-length, ready for PromptBuilder.build_messages().
        """
        question: str = completed_state.get("question", "")
        final_answer: str = completed_state.get("final_answer") or ""
        answer_type: str = completed_state.get("answer_type") or "text"
        existing_history: list[ChatCompletionMessageParam] = list(
            completed_state.get("conversation_history") or []
        )

        # Store the new assistant message with a type prefix on the first line.
        # The prefix survives the verbatim stage so compression can extract it
        # in a future call without needing answer_type to be passed again.
        new_user_msg: ChatCompletionMessageParam = {"role": "user", "content": question}
        new_assistant_msg: ChatCompletionMessageParam = {
            "role": "assistant",
            "content": f"[{answer_type}]\n{final_answer}",
        }

        full_flat = existing_history + [new_user_msg, new_assistant_msg]
        pairs = _to_pairs(full_flat)

        if len(pairs) <= self._max_context_turns:
            return _to_flat(pairs)

        recent_pairs = pairs[-self._max_context_turns:]
        older_pairs = pairs[:-self._max_context_turns]
        compressed_pairs = [_compress_pair(p) for p in older_pairs]

        return _to_flat(compressed_pairs + recent_pairs)