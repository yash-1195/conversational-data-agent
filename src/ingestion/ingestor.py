"""
Data ingestion module.

Handles file upload validation and parsing. Enforces size and row limits
defined in config before any data reaches the agent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.core.config_loader import AppConfig


class IngestionError(Exception):
    """Raised when a file fails validation or cannot be parsed.

    Always carries a user-facing message — never expose a raw traceback
    upstream from this class.
    """
    pass


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class DataIngestor:
    """Validates and loads a dataset file into a pandas DataFrame.

    Enforces the file size and row count limits defined in ``AppConfig.data``.
    On any violation, raises ``IngestionError`` with a clean, readable message.

    Example
    -------
    >>> ingestor = DataIngestor(config)
    >>> df = ingestor.load("/path/to/data.csv")
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, file_path: str | Path) -> pd.DataFrame:
        """Load and validate a dataset file.

        Parameters
        ----------
        file_path:
            Absolute or relative path to a CSV or Excel file.

        Returns
        -------
        pd.DataFrame
            The parsed dataset.

        Raises
        ------
        IngestionError
            If the file does not exist, has an unsupported format, exceeds
            the configured size limit, or exceeds the configured row limit.
        """
        path = Path(file_path)

        self._check_exists(path)
        self._check_extension(path)
        self._check_file_size(path)

        df = self._parse(path)

        self._check_row_count(df, path.name)

        return df

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _check_exists(self, path: Path) -> None:
        if not path.exists():
            raise IngestionError(f"File not found: '{path}'")

    def _check_extension(self, path: Path) -> None:
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise IngestionError(
                f"Unsupported file type '{ext}'. "
                f"Supported formats: {supported}"
            )

    def _check_file_size(self, path: Path) -> None:
        max_bytes = self._config.data.max_upload_size_mb * 1024 * 1024
        actual_bytes = path.stat().st_size
        if actual_bytes > max_bytes:
            actual_mb = actual_bytes / (1024 * 1024)
            raise IngestionError(
                f"File '{path.name}' is {actual_mb:.1f} MB, which exceeds the "
                f"{self._config.data.max_upload_size_mb} MB upload limit. "
                "Please reduce the file size or increase the limit in config.yaml."
            )

    def _check_row_count(self, df: pd.DataFrame, filename: str) -> None:
        row_count = len(df)
        if row_count > self._config.data.max_row_limit:
            raise IngestionError(
                f"'{filename}' contains {row_count:,} rows, which exceeds the "
                f"{self._config.data.max_row_limit:,} row limit. "
                "Please filter the dataset or increase the limit in config.yaml."
            )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self, path: Path) -> pd.DataFrame:
        ext = path.suffix.lower()
        try:
            if ext == ".csv":
                return pd.read_csv(path)
            elif ext == ".xlsx":
                return pd.read_excel(path, engine="openpyxl")
            else:  # .xls
                return pd.read_excel(path, engine="xlrd")
        except Exception as exc:
            raise IngestionError(
                f"Failed to parse '{path.name}': {exc}. "
                "Ensure the file is a valid CSV or Excel document."
            ) from exc