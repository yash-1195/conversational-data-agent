# app.py

"""
Streamlit frontend for the Conversational Data Analysis Agent.

Entry point — run with:
    streamlit run app.py

Layout
------
Sidebar  — file uploader with size/row limit messaging, dataset info summary
Main     — collapsible dataset preview (schema + quality flags), chat history,
           chat input

Session state keys
------------------
initialised          bool              — prevents re-init on every Streamlit rerun
df                   DataFrame | None  — loaded dataset
pickle_path          str | None        — path to pickled df used by the sandbox
profile              DataProfile | None
conversation_history list              — sliding window managed by ContextManager
chat_display         list[dict]        — display log: {role, content, answer_type}
context_manager      ContextManager
temp_dir             str               — temp directory for df pickle (plots go to config.ui.plot_dir)

MLflow logging
--------------
One ExperimentLogger run per user question, created fresh each turn.
Wraps graph.invoke() and logs params, metrics, tags, and the plot artifact
(if applicable) before finalising. Consistent with the project-wide pattern
of keeping MLflow calls outside individual components.
"""

from __future__ import annotations

import pickle
import tempfile
import time
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.agent.context_manager import ContextManager
from src.agent.graph import build_graph
from src.agent.state import make_initial_state
from src.core.config_loader import load_config
from src.core.experiment_logger import ExperimentLogger
from src.core.llm_client import LLMClient, LLMClientError
from src.ingestion.ingestor import DataIngestor, IngestionError
from src.ingestion.profiler import DataProfiler

load_dotenv()

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Conversational Data Agent",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached resources
# Loaded once per Streamlit server process, not on every rerun.
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_resources():
    """
    Load config, build LLMClient, compile graph.

    Called once at startup. Using a single no-argument cached function
    avoids Pydantic model hashing issues that arise when passing config
    as a cache_resource argument.
    """
    config = load_config()
    client = LLMClient(config.llm)
    graph = build_graph(client=client, config=config, plot_dir=config.ui.plot_dir)
    return config, graph


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_session_state(config) -> None:
    """Initialise session state keys on the first rerun of a new browser session."""
    if "initialised" in st.session_state:
        return

    st.session_state.initialised = True
    st.session_state.df = None
    st.session_state.pickle_path = None
    st.session_state.profile = None
    st.session_state.conversation_history = []
    st.session_state.chat_display = []
    st.session_state.context_manager = ContextManager(
        max_context_turns=config.llm.max_context_turns
    )
    # Temp directory persists for the lifetime of the browser session.
    # Both the DataFrame pickle and generated plot PNGs are written here.
    st.session_state.temp_dir = tempfile.mkdtemp()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pickle_dataframe(df, temp_dir: str) -> str:
    """Pickle df to a fixed path inside temp_dir. Returns the absolute path."""
    path = Path(temp_dir) / "dataframe.pkl"
    with open(path, "wb") as f:
        pickle.dump(df, f)
    return str(path.resolve())


def _reset_for_new_upload(config) -> None:
    """
    Clear all dataset-dependent state when a new file is uploaded.

    Resets the DataFrame, profile, pickle path, conversation history,
    and chat display. Does not recreate temp_dir — the same directory
    is reused to avoid leaking temp files across uploads.
    """
    st.session_state.df = None
    st.session_state.pickle_path = None
    st.session_state.profile = None
    st.session_state.conversation_history = []
    st.session_state.chat_display = []
    st.session_state.context_manager = ContextManager(
        max_context_turns=config.llm.max_context_turns
    )


def _render_answer(answer_type: str, content: str) -> None:
    """
    Render an assistant answer using the appropriate Streamlit widget.

    answer_type dispatch:
      "plot"          → st.image (content is absolute PNG path)
      "error"         → st.error
      "table"         → st.markdown (content is a markdown table string)
      "text"          → st.markdown
      "clarification" → st.markdown
    """
    if answer_type == "plot":
        if Path(content).is_file():
            st.image(content)
        else:
            st.warning("Plot image could not be found on disk.")
    elif answer_type == "error":
        st.error(content)
    else:
        st.markdown(content)


def _run_agent(question: str, config, graph) -> dict:
    """
    Build initial state, invoke the graph, and wrap the call with MLflow logging.

    Parameters
    ----------
    question:
        The user's natural language question for this turn.
    config:
        Full AppConfig. Used to source the plot_dir path.
    graph:
        Compiled LangGraph graph returned by build_graph().

    Returns
    -------
    dict
        The completed AgentState after the graph finishes.
    """
    state = make_initial_state(
        question=question,
        dataframe_pickle_path=st.session_state.pickle_path,
        profile=st.session_state.profile,
        conversation_history=st.session_state.conversation_history,
    )

    logger = ExperimentLogger(experiment_name="conversational-data-agent")
    question_id = uuid.uuid4().hex[:8]
    profile = st.session_state.profile

    logger.start_question_run(
        question_id=question_id,
        question_text=question,
        dataset_name=profile.dataset_name,
        row_count=profile.row_count,
        col_count=profile.col_count,
    )

    start_time = time.time()
    result_state = graph.invoke(state)
    latency_ms = (time.time() - start_time) * 1000

    # Log each failed attempt.
    # Note: generated_code in state is the final attempt's code only —
    # intermediate attempt codes are not stored in AgentState. This is a
    # known limitation; best available code is logged for all attempts.
    for attempt_idx, failure_type, retry_instruction in result_state["failure_history"]:
        logger.log_attempt(
            attempt_index=attempt_idx,
            error_type=failure_type,
            error_message=retry_instruction,
            code=result_state["generated_code"],
        )

    # Log plot PNG as MLflow artifact if this turn produced one.
    if result_state.get("plot_path"):
        logger.log_artifact(result_state["plot_path"])

    answer_type = result_state["answer_type"]
    # clarification is a normal outcome (not a failure), so only "error" is false.
    success = answer_type not in ("error", "")

    logger.finalize(
        success=success,
        answer_type=answer_type,
        latency_ms=latency_ms,
        total_attempts=result_state["attempt_count"],
    )

    # Update context window for the next turn.
    st.session_state.conversation_history = (
        st.session_state.context_manager.update(result_state)
    )

    return result_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        config, graph = _get_resources()
    except LLMClientError as exc:
        st.error(
            f"**Startup failed:** {exc}  \n"
            "Check that your API key environment variable is set correctly "
            "(see `configs/config.yaml` → `llm.api_key_env_var`)."
        )
        st.stop()
    _init_session_state(config)

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------

    with st.sidebar:
        st.title("📊 Data Agent")
        st.caption("Upload a dataset and ask questions in plain English.")

        st.divider()

        st.subheader("Upload Dataset")
        st.caption(
            f"Supported: CSV, XLSX, XLS  \n"
            f"Max file size: **{config.data.max_upload_size_mb} MB**  \n"
            f"Max rows: **{config.data.max_row_limit:,}**"
        )

        uploaded_file = st.file_uploader(
            label="Choose a file",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            current_name = (
                st.session_state.profile.dataset_name
                if st.session_state.profile is not None
                else None
            )

            # Only re-ingest when the filename changes — avoids redundant
            # work on every Streamlit rerun while the same file is uploaded.
            if uploaded_file.name != current_name:
                _reset_for_new_upload(config)

                with st.spinner("Loading and profiling dataset…"):
                    # DataIngestor.load() requires a real file path (it
                    # checks size on disk). Write the uploaded bytes to a
                    # temp file before passing it in.
                    tmp_upload_path = (
                        Path(st.session_state.temp_dir) / uploaded_file.name
                    )
                    tmp_upload_path.write_bytes(uploaded_file.getvalue())

                    try:
                        ingestor = DataIngestor(config)
                        df = ingestor.load(tmp_upload_path)

                        profiler = DataProfiler()
                        profile = profiler.profile(
                            df, dataset_name=uploaded_file.name
                        )

                        pickle_path = _pickle_dataframe(
                            df, st.session_state.temp_dir
                        )

                        st.session_state.df = df
                        st.session_state.profile = profile
                        st.session_state.pickle_path = pickle_path

                    except IngestionError as exc:
                        st.error(str(exc))

        # Dataset summary in sidebar (shown once a file is loaded)
        if st.session_state.profile is not None:
            profile = st.session_state.profile
            st.divider()
            st.subheader("Loaded Dataset")
            st.markdown(
                f"**{profile.dataset_name}**  \n"
                f"{profile.row_count:,} rows × {profile.col_count} columns"
            )
            if profile.quality_flags:
                flag_lines = "\n".join(f"• {f}" for f in profile.quality_flags)
                st.warning(f"Dataset quality flags:\n{flag_lines}")

    # ------------------------------------------------------------------
    # Main area — no dataset uploaded
    # ------------------------------------------------------------------

    if st.session_state.profile is None:
        st.title("Conversational Data Analysis Agent")
        st.info(
            "Upload a CSV or Excel file using the sidebar to get started.  \n"
            "Then ask questions about your data in plain English."
        )
        return

    # ------------------------------------------------------------------
    # Main area — dataset loaded
    # ------------------------------------------------------------------

    profile = st.session_state.profile

    # Collapsible dataset preview
    with st.expander(
        f"📋 Dataset Preview — {profile.dataset_name}",
        expanded=False,
    ):
        st.dataframe(
            st.session_state.df.head(config.ui.preview_rows),
            use_container_width=True,
        )

        flagged_cols = profile.flagged_columns()
        if flagged_cols:
            st.markdown("**Column quality flags:**")
            for col_name in flagged_cols:
                flags = profile.get_column_flags(col_name)
                st.markdown(f"⚠ `{col_name}`: {'; '.join(flags)}")
        else:
            st.caption("No column-level quality flags detected.")

    st.divider()

    # Chat history
    display_history = st.session_state.chat_display[
        -config.ui.max_chat_history_display:
    ]
    for entry in display_history:
        with st.chat_message(entry["role"]):
            _render_answer(entry["answer_type"], entry["content"])

    # Chat input
    question = st.chat_input("Ask a question about your data…")

    if question:
        # Append and render the user message immediately so the UI
        # feels responsive while the agent runs.
        st.session_state.chat_display.append(
            {"role": "user", "content": question, "answer_type": "text"}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    result_state = _run_agent(question, config, graph)
                    final_answer = result_state["final_answer"] or ""
                    answer_type = result_state["answer_type"]
                except Exception as exc:
                    # Catch unexpected errors (e.g. LLMClientError from a
                    # connectivity failure) and surface them cleanly rather
                    # than letting Streamlit show a raw traceback.
                    final_answer = (
                        f"An unexpected error occurred: {exc}  \n"
                        "Please check your API key and network connection."
                    )
                    answer_type = "error"

            _render_answer(answer_type, final_answer)

            st.session_state.chat_display.append(
                {
                    "role": "assistant",
                    "content": final_answer,
                    "answer_type": answer_type,
                }
            )


if __name__ == "__main__":
    main()