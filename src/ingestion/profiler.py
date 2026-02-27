"""
Data profiling module.

Generates a structured profile of a DataFrame including schema details,
basic statistics, and data quality flags. The profile is designed to be
injected directly into the LLM system prompt and referenced during response
generation to surface relevant data quality caveats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

import pandas as pd


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class ColumnProfile:
    """Profile information for a single column."""
    name: str
    dtype: str
    null_count: int
    null_rate: float          # 0.0 – 1.0
    unique_count: int
    sample_values: list       # up to 5 representative values
    flags: list[str] = field(default_factory=list)


@dataclass
class DataProfile:
    """Complete profile for a loaded DataFrame.

    Attributes
    ----------
    dataset_name:
        Filename or identifier passed at profile time.
    row_count:
        Number of rows in the dataset.
    col_count:
        Number of columns.
    columns:
        Per-column profiles, in order.
    quality_flags:
        Dataset-level quality issues (e.g. duplicate rows).
    """
    dataset_name: str
    row_count: int
    col_count: int
    columns: list[ColumnProfile]
    quality_flags: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Prompt-ready rendering
    # ------------------------------------------------------------------

    def to_prompt_str(self) -> str:
        """Render the profile as a compact, LLM-readable string.

        This string is injected into the system prompt at the start of each
        agent turn so the model always has fresh schema context.
        """
        lines: list[str] = []
        lines.append(f"=== Dataset: {self.dataset_name} ===")
        lines.append(f"Shape: {self.row_count:,} rows × {self.col_count} columns")

        if self.quality_flags:
            lines.append("\nDataset-level quality flags:")
            for flag in self.quality_flags:
                lines.append(f"  ⚠ {flag}")

        lines.append("\nColumn summary:")
        for col in self.columns:
            flag_str = ""
            if col.flags:
                flag_str = "  [FLAGS: " + "; ".join(col.flags) + "]"
            null_pct = f"{col.null_rate * 100:.1f}%"
            sample = ", ".join(str(v) for v in col.sample_values[:3])
            lines.append(
                f"  {col.name} | {col.dtype} | nulls: {null_pct} | "
                f"unique: {col.unique_count:,} | sample: [{sample}]{flag_str}"
            )

        return "\n".join(lines)

    def flagged_columns(self) -> list[str]:
        """Return names of columns that have at least one quality flag."""
        return [c.name for c in self.columns if c.flags]

    def get_column_flags(self, column_name: str) -> list[str]:
        """Return the quality flags for a specific column, or empty list."""
        for col in self.columns:
            if col.name == column_name:
                return col.flags
        return []


# ------------------------------------------------------------------
# Profiler
# ------------------------------------------------------------------

# Threshold above which a null rate is flagged as high
_HIGH_NULL_THRESHOLD = 0.20

# Patterns that suggest a column might be a date stored as string
_DATE_PATTERNS = [
    re.compile(r"^\d{4}-\d{2}-\d{2}"),          # ISO: 2024-01-15
    re.compile(r"^\d{2}/\d{2}/\d{4}"),           # US: 01/15/2024
    re.compile(r"^\d{2}-\d{2}-\d{4}"),           # EU: 15-01-2024
    re.compile(r"^\d{4}/\d{2}/\d{2}"),           # 2024/01/15
    re.compile(r"^\w+ \d{1,2},?\s+\d{4}"),       # January 15, 2024
]

# Minimum fraction of non-null values that must match a date pattern
_DATE_PATTERN_MIN_FRACTION = 0.80

# Symmetric boundary: flag a column as mixed-type if numeric_frac falls in
# the range (_MIXED_TYPE_MIN_FRACTION, 1 - _MIXED_TYPE_MIN_FRACTION),
# i.e. neither fully numeric nor fully non-numeric.
_MIXED_TYPE_MIN_FRACTION = 0.05


class DataProfiler:
    """Generates a :class:`DataProfile` from a pandas DataFrame.

    Example
    -------
    >>> profiler = DataProfiler()
    >>> profile = profiler.profile(df, dataset_name="sales.csv")
    >>> print(profile.to_prompt_str())
    """

    def profile(
        self,
        df: pd.DataFrame,
        dataset_name: str = "dataset",
        sample_rows: int = 5,
    ) -> DataProfile:
        """Profile a DataFrame.

        Parameters
        ----------
        df:
            The loaded dataset.
        dataset_name:
            Identifier used in the prompt rendering (typically the filename).
        sample_rows:
            Number of representative sample values to include per column.

        Returns
        -------
        DataProfile
        """
        column_profiles = [
            self._profile_column(df, col, sample_rows)
            for col in df.columns
        ]

        dataset_flags = self._dataset_flags(df)

        return DataProfile(
            dataset_name=dataset_name,
            row_count=len(df),
            col_count=len(df.columns),
            columns=column_profiles,
            quality_flags=dataset_flags,
        )

    # ------------------------------------------------------------------
    # Column-level profiling
    # ------------------------------------------------------------------

    def _profile_column(
        self,
        df: pd.DataFrame,
        col: str,
        sample_rows: int,
    ) -> ColumnProfile:
        series = df[col]
        n = len(series)
        null_count = int(series.isna().sum())
        null_rate = null_count / n if n > 0 else 0.0
        unique_count = int(series.nunique(dropna=True))

        # Sample: drop nulls, take first `sample_rows` unique values
        sample_values = (
            series.dropna()
            .drop_duplicates()
            .head(sample_rows)
            .tolist()
        )

        flags: list[str] = []

        if null_rate >= _HIGH_NULL_THRESHOLD:
            flags.append(f"high null rate ({null_rate * 100:.0f}%)")

        if self._is_potential_date_string(series):
            flags.append("possible date column stored as string")

        if self._has_mixed_types(series):
            flags.append("mixed types detected")

        return ColumnProfile(
            name=col,
            dtype=str(series.dtype),
            null_count=null_count,
            null_rate=null_rate,
            unique_count=unique_count,
            sample_values=sample_values,
            flags=flags,
        )

    # ------------------------------------------------------------------
    # Quality flag detection helpers
    # ------------------------------------------------------------------

    def _is_potential_date_string(self, series: pd.Series) -> bool:
        """Return True if the column is object/string dtype and most values
        look like dates."""
        if series.dtype != object:
            return False

        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            return False

        # Check each pattern against the first 200 values (perf guard)
        sample = non_null.head(200)
        for pattern in _DATE_PATTERNS:
            matches = sample.str.match(pattern).sum()
            if matches / len(sample) >= _DATE_PATTERN_MIN_FRACTION:
                return True

        return False

    def _has_mixed_types(self, series: pd.Series) -> bool:
        """Return True if an object-dtype column contains a meaningful mix
        of numeric and non-numeric values."""
        if series.dtype != object:
            return False

        non_null = series.dropna()
        if len(non_null) == 0:
            return False

        numeric_mask = pd.to_numeric(non_null, errors="coerce").notna()
        numeric_frac = numeric_mask.sum() / len(non_null)

        # Flag if neither fully numeric nor fully non-numeric
        return _MIXED_TYPE_MIN_FRACTION < numeric_frac < (1.0 - _MIXED_TYPE_MIN_FRACTION)

    # ------------------------------------------------------------------
    # Dataset-level flags
    # ------------------------------------------------------------------

    def _dataset_flags(self, df: pd.DataFrame) -> list[str]:
        flags: list[str] = []

        # Duplicate rows
        dup_count = int(df.duplicated().sum())
        if dup_count > 0 and len(df) > 0:
            dup_pct = dup_count / len(df) * 100
            flags.append(f"{dup_count:,} duplicate rows ({dup_pct:.1f}%)")

        # Entirely empty columns
        empty_cols = [col for col in df.columns if df[col].isna().all()]
        if empty_cols:
            flags.append(f"entirely null columns: {', '.join(empty_cols)}")

        return flags