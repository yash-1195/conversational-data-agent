# src/agent/output_formatter.py

"""
OutputFormatter — dispatches an execution result to the correct rendering
path and appends data quality caveats where relevant.

Called by the `respond` node in graph.py on the success path (Phase 3).
Replaces the str(result) placeholder introduced in Phase 2.

Result type dispatch
--------------------
  matplotlib Figure  →  saved as PNG, answer_type = "plot"
  plotly Figure      →  saved as PNG, answer_type = "plot"
  pd.DataFrame       →  markdown table string, answer_type = "table"
  pd.Series          →  promoted to single-column DataFrame, answer_type = "table"
  numpy scalar       →  Python scalar via .item(), answer_type = "text"
  int / float / bool →  str(), answer_type = "text"
  str                →  returned as-is, answer_type = "text"
  anything else      →  str() fallback, answer_type = "text"

Data quality caveats
--------------------
For DataFrame results: column-level flags are checked for every column
that appears in the result, then dataset-level flags are appended.

For all other result types: only dataset-level flags are surfaced, because
the formatter cannot know which columns the LLM used to produce a scalar
or plot result.

MLflow logging
--------------
Intentionally excluded. The caller (respond node → graph caller) logs
plot_path to MLflow as an artifact after inspecting the returned
FormattedOutput. This is consistent with the project-wide pattern of
keeping MLflow calls outside individual components.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
import numpy as np

from src.ingestion.profiler import DataProfile


# Optional dependency guards — both libraries are in requirements-base.txt
# but we import defensively so unit tests can mock or skip these paths.
try:
    import matplotlib.figure as _mpl_figure
    _MATPLOTLIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MATPLOTLIB_AVAILABLE = False

try:
    import plotly.basedatatypes as _plotly_base
    _PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PLOTLY_AVAILABLE = False


# Maximum rows included in a markdown table before truncation notice is appended.
# Large tables sent through final_answer → Streamlit would otherwise be huge
# strings that slow rendering. The full DataFrame is available in session state.
_TABLE_MAX_ROWS = 50


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

@dataclass
class FormattedOutput:
    """
    Result of formatting a single agent turn's execution output.

    Attributes
    ----------
    answer_type:
        One of "table", "plot", "text". Written to AgentState.answer_type
        so the frontend and ContextManager know how to render/compress it.
    content:
        The displayable answer string:
          - "table" → markdown table (truncated to _TABLE_MAX_ROWS rows)
          - "plot"  → absolute path to the saved PNG file
          - "text"  → plain string representation of the result value
    caveat:
        Data quality warning string, or empty string if nothing was flagged.
        The respond node appends this to final_answer before returning.
    plot_path:
        Absolute path to the saved PNG. Populated only for "plot" results;
        None for "table" and "text". Provided as a separate field so the
        caller can log it to MLflow without re-parsing content.
    """
    answer_type: Literal["table", "plot", "text"]
    content: str
    caveat: str
    plot_path: Optional[str]


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class OutputFormatter:
    """
    Dispatches a result value to the correct rendering path and builds
    any relevant data quality caveat.

    Stateless — safe to instantiate once in GraphNodes and reuse across
    every turn in a session.

    Usage
    -----
        formatter = OutputFormatter()
        output = formatter.format(
            result=execution_result.result,
            profile=state["profile"],
            plot_dir="outputs/plots",
        )
        # output.content  → final answer string (or image path for plots)
        # output.caveat   → append to final_answer if non-empty
        # output.plot_path → log to MLflow as artifact if non-None
    """

    def format(
        self,
        result: Any,
        profile: DataProfile,
        plot_dir: str = "outputs/plots",
    ) -> FormattedOutput:
        """
        Dispatch result to the correct rendering path.

        Parameters
        ----------
        result:
            The value assigned to `result` in the executed user code.
            Must not be None or empty — OutputValidator already rejects
            those cases before the respond node is reached.
        profile:
            DataProfile for the active dataset. Used to build caveats.
        plot_dir:
            Directory where plot PNGs are saved. Created if it does not
            exist. Defaults to "outputs/plots" relative to the working
            directory. Pass an absolute path for deterministic output
            locations in tests or production.

        Returns
        -------
        FormattedOutput
        """
        # Check matplotlib first — plotly figures are not mpl figures,
        # so the order here does not cause cross-contamination.
        if _MATPLOTLIB_AVAILABLE and isinstance(result, _mpl_figure.Figure):
            return self._format_matplotlib(result, profile, plot_dir)

        if _PLOTLY_AVAILABLE and isinstance(result, _plotly_base.BaseFigure):
            return self._format_plotly(result, profile, plot_dir)

        if isinstance(result, pd.DataFrame):
            return self._format_dataframe(result, profile)

        # Series: promote to single-column DataFrame for uniform table rendering.
        if isinstance(result, pd.Series):
            return self._format_dataframe(result.to_frame(), profile)

        # numpy scalar: extract the underlying Python value first so that
        # str() on a numpy int64 produces "42" not "np.int64(42)".
        if isinstance(result, np.generic):
            return self._format_text(result.item(), profile)

        if isinstance(result, (int, float, bool, str)):
            return self._format_text(result, profile)

        # list, dict, or any other unexpected type.
        return self._format_text(result, profile)

    # ------------------------------------------------------------------
    # Rendering paths
    # ------------------------------------------------------------------

    def _format_dataframe(
        self,
        df: pd.DataFrame,
        profile: DataProfile,
    ) -> FormattedOutput:
        truncated = len(df) > _TABLE_MAX_ROWS
        display_df = df.head(_TABLE_MAX_ROWS)

        try:
            # to_markdown() requires the `tabulate` package. It produces the
            # cleanest output for Streamlit's st.markdown(), so we prefer it.
            content = display_df.to_markdown(index=False)
        except ImportError:
            # tabulate not installed — fall back to pandas plain string.
            content = display_df.to_string(index=False)

        if truncated:
            content += (
                f"\n\n_(Showing first {_TABLE_MAX_ROWS:,} of {len(df):,} rows.)_"
            )

        caveat = self._build_dataframe_caveat(df, profile)
        return FormattedOutput(
            answer_type="table",
            content=content,
            caveat=caveat,
            plot_path=None,
        )

    def _format_matplotlib(
        self,
        fig: Any,
        profile: DataProfile,
        plot_dir: str,
    ) -> FormattedOutput:
        path = self._save_png_matplotlib(fig, plot_dir)

        # Close the figure immediately to free memory. The image is now on disk.
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass

        caveat = self._build_dataset_level_caveat(profile)
        return FormattedOutput(
            answer_type="plot",
            content=path,
            caveat=caveat,
            plot_path=path,
        )

    def _format_plotly(
        self,
        fig: Any,
        profile: DataProfile,
        plot_dir: str,
    ) -> FormattedOutput:
        path = self._save_png_plotly(fig, plot_dir)
        caveat = self._build_dataset_level_caveat(profile)
        return FormattedOutput(
            answer_type="plot",
            content=path,
            caveat=caveat,
            plot_path=path,
        )

    def _format_text(
        self,
        value: Any,
        profile: DataProfile,
    ) -> FormattedOutput:
        caveat = self._build_dataset_level_caveat(profile)
        return FormattedOutput(
            answer_type="text",
            content=str(value),
            caveat=caveat,
            plot_path=None,
        )

    # ------------------------------------------------------------------
    # Plot saving
    # ------------------------------------------------------------------

    def _save_png_matplotlib(self, fig: Any, plot_dir: str) -> str:
        """Save a matplotlib Figure to plot_dir as PNG. Returns absolute path."""
        out_dir = Path(plot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"plot_{uuid.uuid4().hex[:8]}.png"
        fig.savefig(str(path), bbox_inches="tight", dpi=150)
        return str(path.resolve())

    def _save_png_plotly(self, fig: Any, plot_dir: str) -> str:
        """Save a plotly Figure to plot_dir as PNG. Returns absolute path.

        Requires the `kaleido` package (`pip install kaleido`) for static
        image export. If kaleido is missing, write_image() will raise a
        ValueError with a clear message — no special handling needed here.
        """
        out_dir = Path(plot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"plot_{uuid.uuid4().hex[:8]}.png"
        fig.write_image(str(path))
        return str(path.resolve())

    # ------------------------------------------------------------------
    # Caveat builders
    # ------------------------------------------------------------------

    def _build_dataframe_caveat(
        self,
        df: pd.DataFrame,
        profile: DataProfile,
    ) -> str:
        """
        Build a caveat string for a DataFrame result.

        Checks in order:
        1. Column-level flags for any column present in the result.
        2. Dataset-level quality flags.

        Returns an empty string if nothing is flagged.
        """
        lines: list[str] = []

        # Intersect result columns with columns the profiler flagged.
        result_cols = set(df.columns.tolist())
        flagged = set(profile.flagged_columns())
        relevant = sorted(result_cols & flagged)

        for col_name in relevant:
            col_flags = profile.get_column_flags(col_name)
            if col_flags:
                flag_str = "; ".join(col_flags)
                lines.append(f"  • '{col_name}': {flag_str}")

        for flag in profile.quality_flags:
            lines.append(f"  • {flag}")

        if not lines:
            return ""

        return "⚠ Data quality note:\n" + "\n".join(lines)

    def _build_dataset_level_caveat(self, profile: DataProfile) -> str:
        """
        Build a caveat string for non-DataFrame results (scalars, plots).

        Only surfaces dataset-level flags. Column-level flags are omitted
        because the formatter cannot know which columns contributed to a
        scalar value or a plot.
        """
        if not profile.quality_flags:
            return ""

        lines = [f"  • {flag}" for flag in profile.quality_flags]
        return "⚠ Data quality note:\n" + "\n".join(lines)