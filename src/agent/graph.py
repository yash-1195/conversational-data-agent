"""
Agent graph — wires the four LangGraph nodes into a compiled executable graph.

Topology
--------

    ┌─────────────────────────────────────────────┐
    │                   plan                      │
    └──────────────┬──────────────────────────────┘
                   │
          needs_clarification?
          ┌────────┴────────┐
         yes               no
          │                 │
          ▼                 ▼
       respond           execute
          ▲                 │
          │                 ▼
          │             evaluate
          │                 │
          │     should_retry AND attempts_remaining?
          │         ┌───────┴───────┐
          │        yes              no
          │         │               │
          └─────────┘               ▼
          (retry loop)           respond ──► END

Nodes
-----
plan      — checks for ambiguity (ClarificationTool), then calls the LLM
            to generate code. On retry, the system prompt includes the
            previous failure reason via PromptBuilder.
execute   — runs the generated code in the sandboxed subprocess
            (CodeExecutionTool).
evaluate  — validates the execution result (OutputValidator). On failure,
            increments attempt_count and appends to failure_history before
            routing back to plan.
respond   — formats the final answer. Handles three paths:
              1. clarification  — surfaces clarifying_question to the user
              2. error          — formats the exhausted-retry error message
              3. success        — dispatches result to OutputFormatter:
                   DataFrame/Series → markdown table
                   matplotlib/plotly Figure → saved PNG path
                   scalar/str/numpy scalar → plain text
                 Data-quality caveats are appended when relevant.

MLflow logging
--------------
Intentionally excluded from graph nodes. The caller wraps graph.invoke()
and has access to the completed AgentState, which contains everything
ExperimentLogger needs: answer_type, attempt_count, failure_history,
final_answer. See ExperimentLogger.start_question_run() / finalize().

Usage
-----
    from src.core.config_loader import load_config
    from src.core.llm_client import LLMClient
    from src.agent.graph import build_graph
    from src.agent.state import make_initial_state

    config = load_config()
    client = LLMClient(config.llm)
    graph  = build_graph(client=client, config=config)

    state  = make_initial_state(
        question="What is total revenue by region?",
        dataframe_pickle_path="/tmp/df.pkl",
        profile=profile,
        conversation_history=previous_history,
    )
    result_state = graph.invoke(state)
    final_answer = result_state["final_answer"]
"""

from __future__ import annotations

import re

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.clarification import ClarificationTool
from src.agent.prompt_builder import PromptBuilder
from src.agent.state import AgentState
from src.agent.tools.code_executor import execute_code
from src.agent.output_formatter import OutputFormatter
from src.agent.validators.output_validator import OutputValidator
from src.core.config_loader import AppConfig
from src.core.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Code fence stripping
# ---------------------------------------------------------------------------

_FENCE_PATTERN = re.compile(
    r"^```[\w]*\n(.*?)\n```\s*$",
    re.DOTALL,
)


def _strip_code_fences(text: str) -> str:
    """
    Remove markdown code fences that LLMs insert despite prompt instructions.

    Handles:
      - ```python\n...\n```
      - ```\n...\n```

    If no fences are present the text is returned unchanged.
    """
    match = _FENCE_PATTERN.match(text.strip())
    return match.group(1) if match else text


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def _route_from_plan(state: AgentState) -> str:
    """
    After plan: if ambiguity was detected skip straight to respond,
    otherwise proceed to execute.
    """
    return "respond" if state["needs_clarification"] else "execute"


def _make_route_from_evaluate(config: AppConfig):
    """
    Return the conditional edge function for evaluate → plan/respond.

    Extracted from GraphNodes so it can be tested without constructing a
    full GraphNodes instance (which requires a live LLMClient).

    Routes back to plan if the outcome failed AND there are attempts
    remaining. Otherwise routes to respond (success or exhausted retries).
    Uses attempt_count *after* evaluate has already incremented it on
    failure — so once it reaches the cap, retrying stops.
    """
    def _route(state: AgentState) -> str:
        outcome = state["validation_outcome"]
        if outcome is None or not outcome.should_retry:
            return "respond"
        if state["attempt_count"] >= config.agent.max_plan_attempts:
            return "respond"
        return "plan"
    return _route


# ---------------------------------------------------------------------------
# Graph nodes (class-based for dependency injection)
# ---------------------------------------------------------------------------

class GraphNodes:
    """
    Holds the four node methods and the shared dependencies they need.

    One instance is created by build_graph() and its methods are registered
    as LangGraph node callables. This avoids closures for each node and
    makes the class unit-testable by injecting mock dependencies.

    Parameters
    ----------
    client:
        Initialised LLMClient. Shared across all plan-node calls.
    config:
        Full AppConfig. Nodes read execution.timeout_s and
        agent.max_plan_attempts from it.
    """

    def __init__(self, client: LLMClient, config: AppConfig) -> None:
        self._client = client
        self._config = config
        self._builder = PromptBuilder()
        self._clarification_tool = ClarificationTool()
        self._validator = OutputValidator()
        self._formatter = OutputFormatter()

    # ------------------------------------------------------------------
    # Node: plan
    # ------------------------------------------------------------------

    def plan(self, state: AgentState) -> dict:
        """
        1. Run ClarificationTool — if ambiguity detected, set
           needs_clarification and return immediately (graph routes to respond).
        2. Build system prompt and messages via PromptBuilder.
        3. Call LLM and return generated_code.

        On retry attempts, the system prompt includes a retry context section
        with the previous failure reason (built by PromptBuilder automatically
        when attempt_count > 0 and failure_history is non-empty).

        LLMClientError is re-raised and propagates to the caller. This is
        intentional — an LLM connectivity failure is not recoverable within
        the retry loop.
        """
        question = state["question"]
        profile = state["profile"]

        # Step 1: ambiguity check (runs before any LLM call)
        clarification = self._clarification_tool.check(question, profile)
        if clarification.needs_clarification:
            return {
                "needs_clarification": True,
                "clarifying_question": clarification.clarifying_question,
            }

        # Step 2: build prompt
        system_prompt = self._builder.build_system_prompt(
            profile=profile,
            attempt_count=state["attempt_count"],
            failure_history=state["failure_history"],
        )
        messages = self._builder.build_messages(
            question=question,
            conversation_history=state["conversation_history"],
        )

        # Step 3: call LLM — raises LLMClientError on connectivity/auth failure.
        # _strip_code_fences removes markdown fences (```python ... ```) that
        # LLMs often insert despite instructions, which would cause a SyntaxError
        # in the sandbox and burn a retry attempt needlessly.
        raw = self._client.call(system_prompt=system_prompt, messages=messages)
        code = _strip_code_fences(raw)

        return {
            "needs_clarification": False,
            "generated_code": code,
        }

    # ------------------------------------------------------------------
    # Node: execute
    # ------------------------------------------------------------------

    def execute(self, state: AgentState) -> dict:
        """
        Run the generated code in the sandboxed subprocess.

        Passes timeout from config so the sandbox honours the same limit
        as defined in configs/config.yaml rather than the default.
        """
        execution_result = execute_code(
            code=state["generated_code"],
            dataframe_pickle_path=state["dataframe_pickle_path"],
            timeout_s=self._config.execution.timeout_s,
        )
        return {"execution_result": execution_result}

    # ------------------------------------------------------------------
    # Node: evaluate
    # ------------------------------------------------------------------

    def evaluate(self, state: AgentState) -> dict:
        """
        Run OutputValidator against the execution result.

        On failure, increments attempt_count and appends the failure to
        failure_history so the plan node has accurate context on the next
        attempt. These updates happen here (not in plan) so the retry edge
        routing function can read the updated attempt_count.
        """
        execution_result = state["execution_result"]
        if execution_result is None:
            raise RuntimeError(
                "evaluate node reached with no execution_result — "
                "execute must run before evaluate"
            )

        outcome = self._validator.validate(
            execution_result=execution_result,
            question=state["question"],
        )

        updates: dict = {"validation_outcome": outcome}

        if not outcome.passed:
            if outcome.failure_type is None:
                raise RuntimeError(
                    "ValidationOutcome.passed is False but failure_type is None"
                )
            current_attempt = state["attempt_count"]
            updates["attempt_count"] = current_attempt + 1
            updates["failure_history"] = list(state["failure_history"]) + [
                (
                    current_attempt,
                    outcome.failure_type.value,
                    outcome.retry_instruction,
                )
            ]

        return updates

    # ------------------------------------------------------------------
    # Node: respond
    # ------------------------------------------------------------------

    def respond(self, state: AgentState) -> dict:
        """
        Format the final answer and set answer_type on state.

        Three paths:
          1. Clarification — the plan node detected ambiguity.
             final_answer = the clarifying question string.
          2. Error — retries exhausted without a valid result.
             final_answer = the last ValidationOutcome user_message.
          3. Success — execution produced a valid result.
             OutputFormatter dispatches by result type:
               DataFrame/Series  → markdown table, answer_type = "table"
               matplotlib/plotly  → PNG path,        answer_type = "plot"
               scalar/str/numpy  → plain text,       answer_type = "text"
             Data-quality caveats are appended when relevant.
             plot_path is set for plot results so the caller can log
             the PNG artifact to MLflow without re-parsing final_answer.
        """
        # Path 1: clarification
        if state["needs_clarification"]:
            return {
                "final_answer": state["clarifying_question"],
                "answer_type": "clarification",
                "plot_path": None,
            }

        # Path 2: error (retries exhausted)
        outcome = state["validation_outcome"]
        if outcome is not None and not outcome.passed:
            return {
                "final_answer": (
                    f"{outcome.user_message}\n\n"
                    "I was unable to produce a valid answer after "
                    f"{state['attempt_count']} attempt(s). "
                    "Please try rephrasing your question."
                ),
                "answer_type": "error",
                "plot_path": None,
            }

        # Path 3: success
        execution_result = state["execution_result"]
        if execution_result is None:
            raise RuntimeError(
                "respond reached success path but execution_result is None"
            )

        output = self._formatter.format(
            result=execution_result.result,
            profile=state["profile"],
            plot_dir="outputs/plots",
        )

        final_answer = output.content
        if output.caveat:
            final_answer = final_answer + "\n\n" + output.caveat

        return {
            "final_answer": final_answer,
            "answer_type": output.answer_type,
            "plot_path": output.plot_path,
        }


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def build_graph(client: LLMClient, config: AppConfig) -> CompiledStateGraph:
    """
    Build and compile the agent StateGraph.

    Call once at application startup. The returned compiled graph is
    thread-safe and can be reused across all user questions in a session.

    Parameters
    ----------
    client:
        Initialised LLMClient instance.
    config:
        Full AppConfig loaded from configs/config.yaml.

    Returns
    -------
    CompiledStateGraph
        The compiled LangGraph graph. Call .invoke(state) to run it.
    """
    nodes = GraphNodes(client=client, config=config)

    workflow: StateGraph = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("plan", nodes.plan)
    workflow.add_node("execute", nodes.execute)
    workflow.add_node("evaluate", nodes.evaluate)
    workflow.add_node("respond", nodes.respond)

    # Entry point
    workflow.set_entry_point("plan")

    # plan → execute or respond (clarification shortcut)
    workflow.add_conditional_edges(
        "plan",
        _route_from_plan,
        {"execute": "execute", "respond": "respond"},
    )

    # execute → evaluate (always)
    workflow.add_edge("execute", "evaluate")

    # evaluate → plan (retry) or respond (success / exhausted)
    workflow.add_conditional_edges(
        "evaluate",
        _make_route_from_evaluate(config),
        {"plan": "plan", "respond": "respond"},
    )

    # respond → END (always)
    workflow.add_edge("respond", END)

    return workflow.compile()