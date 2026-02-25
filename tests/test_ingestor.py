"""Tests for src/ingestion/ingestor.py"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.ingestion.ingestor import DataIngestor, IngestionError
from src.core.config_loader import (
    AppConfig, DataConfig, LLMConfig, AgentConfig,
    ExecutionConfig, LoggingConfig, Verbosity, OnViolation,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_config(
    max_upload_size_mb: int = 50,
    max_row_limit: int = 500_000,
) -> AppConfig:
    """Build a minimal AppConfig for testing."""
    return AppConfig(
        llm=LLMConfig(
            model_name="gpt-4o",
            temperature=0.0,
            max_tokens=2048,
            max_tool_retries=3,
            max_context_turns=10,
        ),
        agent=AgentConfig(max_plan_attempts=3),
        execution=ExecutionConfig(timeout_s=30),
        logging=LoggingConfig(verbosity=Verbosity.INFO, log_to_file=False),
        data=DataConfig(
            max_upload_size_mb=max_upload_size_mb,
            max_row_limit=max_row_limit,
            on_violation=OnViolation.REJECT,
        ),
    )


def _write_csv(path: Path, rows: int = 5) -> None:
    df = pd.DataFrame({"a": range(rows), "b": range(rows)})
    df.to_csv(path, index=False)


def _write_excel(path: Path, rows: int = 5) -> None:
    df = pd.DataFrame({"x": range(rows), "y": range(rows)})
    df.to_excel(path, index=False)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestDataIngestorCSV:
    def test_load_csv_returns_dataframe(self, tmp_path):
        p = tmp_path / "data.csv"
        _write_csv(p)
        ingestor = DataIngestor(_make_config())
        df = ingestor.load(p)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ["a", "b"]

    def test_load_excel_returns_dataframe(self, tmp_path):
        p = tmp_path / "data.xlsx"
        _write_excel(p)
        ingestor = DataIngestor(_make_config())
        df = ingestor.load(p)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_accepts_string_path(self, tmp_path):
        p = tmp_path / "data.csv"
        _write_csv(p)
        ingestor = DataIngestor(_make_config())
        df = ingestor.load(str(p))
        assert len(df) == 5


class TestDataIngestorValidation:
    def test_raises_on_missing_file(self, tmp_path):
        ingestor = DataIngestor(_make_config())
        with pytest.raises(IngestionError, match="File not found"):
            ingestor.load(tmp_path / "nonexistent.csv")

    def test_raises_on_unsupported_extension(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text('{"a": 1}')
        ingestor = DataIngestor(_make_config())
        with pytest.raises(IngestionError, match="Unsupported file type"):
            ingestor.load(p)

    def test_raises_on_oversized_file(self, tmp_path):
        p = tmp_path / "big.csv"
        _write_csv(p, rows=10)
        ingestor = DataIngestor(_make_config(max_upload_size_mb=1))
        # Mock stat to report a size larger than the 1 MB limit
        mock_stat = MagicMock()
        mock_stat.st_size = 10 * 1024 * 1024  # 10 MB
        with patch.object(Path, "stat", return_value=mock_stat):
            with pytest.raises(IngestionError, match="exceeds the"):
                ingestor.load(p)

    def test_raises_on_row_limit_exceeded(self, tmp_path):
        p = tmp_path / "data.csv"
        _write_csv(p, rows=100)
        ingestor = DataIngestor(_make_config(max_row_limit=10))
        with pytest.raises(IngestionError, match="row limit"):
            ingestor.load(p)

    def test_error_message_includes_filename(self, tmp_path):
        p = tmp_path / "mydata.csv"
        _write_csv(p, rows=50)
        ingestor = DataIngestor(_make_config(max_row_limit=5))
        with pytest.raises(IngestionError) as exc_info:
            ingestor.load(p)
        assert "mydata.csv" in str(exc_info.value)

    def test_error_message_no_traceback_text(self, tmp_path):
        """User-facing errors should not contain Python traceback language."""
        p = tmp_path / "data.json"
        p.write_text("{}")
        ingestor = DataIngestor(_make_config())
        with pytest.raises(IngestionError) as exc_info:
            ingestor.load(p)
        msg = str(exc_info.value)
        assert "Traceback" not in msg
        assert "Exception" not in msg

    def test_raises_on_corrupt_csv(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_bytes(b"\xff\xfe" * 1000)  # binary garbage
        ingestor = DataIngestor(_make_config())
        with pytest.raises(IngestionError, match="Failed to parse"):
            ingestor.load(p)

    def test_exact_row_limit_is_accepted(self, tmp_path):
        """A file with exactly max_row_limit rows should not raise."""
        p = tmp_path / "data.csv"
        _write_csv(p, rows=10)
        ingestor = DataIngestor(_make_config(max_row_limit=10))
        df = ingestor.load(p)
        assert len(df) == 10