# tests/test_code_executor.py

"""
Tests for src/agent/tools/code_executor.py

All tests are self-contained — no external files needed.
DataFrames are constructed inline and pickled to pytest's tmp_path fixture.
"""

import pickle
from pathlib import Path

import pandas as pd
import pytest

from src.agent.tools.code_executor import (
    ALLOWED_MODULES,
    ExecutionResult,
    ExecutionStatus,
    _check_forbidden_imports,
    execute_code,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_df(tmp_path: Path, df: pd.DataFrame) -> str:
    """Pickle a DataFrame to a temp file and return the path as a string."""
    p = tmp_path / "df.pkl"
    with open(p, "wb") as f:
        pickle.dump(df, f)
    return str(p)


def _simple_df(tmp_path: Path) -> str:
    """A minimal 3-row DataFrame for tests that don't care about data shape."""
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie"],
        "score": [85, 92, 78],
        "category": ["A", "B", "A"],
    })
    return _write_df(tmp_path, df)


# ---------------------------------------------------------------------------
# _check_forbidden_imports (pure unit tests — no subprocess)
# ---------------------------------------------------------------------------

class TestCheckForbiddenImports:

    def test_clean_code_returns_none(self):
        code = "import pandas as pd\nresult = df.head()"
        assert _check_forbidden_imports(code) is None

    def test_detects_import_os(self):
        code = "import os\nresult = os.getcwd()"
        assert _check_forbidden_imports(code) == "os"

    def test_detects_from_os_import(self):
        code = "from os import path\nresult = path.exists('.')"
        assert _check_forbidden_imports(code) == "os"

    def test_detects_import_sys(self):
        code = "import sys\nresult = sys.version"
        assert _check_forbidden_imports(code) == "sys"

    def test_detects_import_subprocess(self):
        code = "import subprocess\nresult = subprocess.run(['ls'])"
        assert _check_forbidden_imports(code) == "subprocess"

    def test_detects_import_socket(self):
        code = "import socket\nresult = socket.gethostname()"
        assert _check_forbidden_imports(code) == "socket"

    def test_detects_submodule_import(self):
        # os.path is still os — root module is what matters
        code = "import os.path\nresult = os.path.exists('.')"
        assert _check_forbidden_imports(code) == "os.path"

    def test_allows_numpy(self):
        code = "import numpy as np\nresult = np.mean(df['score'])"
        assert _check_forbidden_imports(code) is None

    def test_allows_matplotlib(self):
        code = "import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\nresult = fig"
        assert _check_forbidden_imports(code) is None

    def test_syntax_error_returns_none(self):
        # Syntax errors are caught at runtime, not by the AST checker
        code = "def broken(:\n    pass"
        assert _check_forbidden_imports(code) is None

    def test_all_forbidden_modules_detected(self):
        """Any module not in ALLOWED_MODULES should be caught."""
        unlisted = ["os", "sys", "subprocess", "socket", "ctypes", "shutil", "requests"]
        for module in unlisted:
            code = f"import {module}\nresult = None"
            result = _check_forbidden_imports(code)
            assert result is not None, f"Expected '{module}' to be caught but wasn't"

    def test_detects_exec_call(self):
        code = "exec('import os'); result = None"
        assert _check_forbidden_imports(code) is not None

    def test_detects_eval_call(self):
        code = "result = eval('1 + 1')"
        assert _check_forbidden_imports(code) is not None

    def test_detects_dunder_import(self):
        code = "result = __import__('os').getcwd()"
        assert _check_forbidden_imports(code) is not None

    def test_allowed_module_not_flagged(self):
        code = "import re\nresult = re.findall(r'\\d+', '123abc')"
        assert _check_forbidden_imports(code) is None


# ---------------------------------------------------------------------------
# execute_code — SUCCESS paths
# ---------------------------------------------------------------------------

class TestExecuteCodeSuccess:

    def test_returns_scalar(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "result = len(df)"
        res = execute_code(code, df_path)
        assert res.status == ExecutionStatus.SUCCESS
        assert res.result == 3

    def test_returns_dataframe(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "result = df[df['category'] == 'A']"
        res = execute_code(code, df_path)
        assert res.status == ExecutionStatus.SUCCESS
        assert isinstance(res.result, pd.DataFrame)
        assert len(res.result) == 2

    def test_returns_string(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "result = df['name'].iloc[0]"
        res = execute_code(code, df_path)
        assert res.status == ExecutionStatus.SUCCESS
        assert res.result == "Alice"

    def test_returns_list(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "result = df['score'].tolist()"
        res = execute_code(code, df_path)
        assert res.status == ExecutionStatus.SUCCESS
        assert res.result == [85, 92, 78]

    def test_groupby_aggregation(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "result = df.groupby('category')['score'].mean().to_dict()"
        res = execute_code(code, df_path)
        assert res.status == ExecutionStatus.SUCCESS
        assert isinstance(res.result, dict)
        assert "A" in res.result

    def test_error_message_empty_on_success(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "result = df.shape"
        res = execute_code(code, df_path)
        assert res.status == ExecutionStatus.SUCCESS
        assert res.error_message == ""
        assert res.retry_instruction == ""

    def test_df_is_available_in_scope(self, tmp_path):
        """The DataFrame must be pre-loaded as `df` in the execution scope."""
        df_path = _simple_df(tmp_path)
        code = "result = list(df.columns)"
        res = execute_code(code, df_path)
        assert res.status == ExecutionStatus.SUCCESS
        assert "name" in res.result
        assert "score" in res.result

    def test_dict_literal_in_code(self, tmp_path):
        """Curly braces in user code must not corrupt the runner script."""
        df_path = _simple_df(tmp_path)
        code = "mapping = {'A': 1, 'B': 2}\nresult = df['category'].map(mapping).tolist()"
        res = execute_code(code, df_path)
        assert res.status == ExecutionStatus.SUCCESS
        assert res.result == [1, 2, 1]

    def test_fstring_in_code(self, tmp_path):
        """f-strings in user code must not corrupt the runner script."""
        df_path = _simple_df(tmp_path)
        code = "result = f\"Row count: {len(df)}\""
        res = execute_code(code, df_path)
        assert res.status == ExecutionStatus.SUCCESS
        assert res.result == "Row count: 3"


# ---------------------------------------------------------------------------
# execute_code — FORBIDDEN_IMPORT
# ---------------------------------------------------------------------------

class TestExecuteCodeForbiddenImport:

    def test_forbidden_import_os(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "import os\nresult = os.getcwd()"
        res = execute_code(code, df_path)
        assert res.status == ExecutionStatus.FORBIDDEN_IMPORT

    def test_forbidden_module_field_populated(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "import sys\nresult = sys.version"
        res = execute_code(code, df_path)
        assert res.forbidden_module == "sys"

    def test_forbidden_error_message_is_user_safe(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "import socket\nresult = socket.gethostname()"
        res = execute_code(code, df_path)
        assert "socket" in res.error_message
        assert "permitted" in res.error_message
        # Must not expose internal paths or stack traces
        assert "Traceback" not in res.error_message
        assert "/tmp" not in res.error_message

    def test_forbidden_retry_instruction_is_actionable(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "import subprocess\nresult = None"
        res = execute_code(code, df_path)
        assert "subprocess" in res.retry_instruction
        assert len(res.retry_instruction) > 20  # not a one-word placeholder

    def test_forbidden_import_caught_before_subprocess(self, tmp_path):
        """AST check fires before any subprocess is spawned — result is instant."""
        df_path = _simple_df(tmp_path)
        code = "import os\nresult = os.listdir('.')"
        # If this completes in well under a second, the subprocess was never spawned.
        # We can't assert timing precisely, but we can assert no subprocess side effects.
        res = execute_code(code, df_path)
        assert res.status == ExecutionStatus.FORBIDDEN_IMPORT
        assert res.stdout == ""  # no subprocess output


# ---------------------------------------------------------------------------
# execute_code — MISSING_RESULT
# ---------------------------------------------------------------------------

class TestExecuteCodeMissingResult:

    def test_no_result_variable(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "x = df.shape"  # assigns to x, not result
        res = execute_code(code, df_path)
        assert res.status == ExecutionStatus.MISSING_RESULT

    def test_missing_result_error_message(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "answer = len(df)"
        res = execute_code(code, df_path)
        assert "result" in res.error_message.lower()

    def test_missing_result_retry_instruction_mentions_result(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "total = df['score'].sum()"
        res = execute_code(code, df_path)
        assert "result" in res.retry_instruction
        assert len(res.retry_instruction) > 20

    def test_empty_code_is_missing_result(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "# just a comment"
        res = execute_code(code, df_path)
        assert res.status == ExecutionStatus.MISSING_RESULT


# ---------------------------------------------------------------------------
# execute_code — RUNTIME_ERROR
# ---------------------------------------------------------------------------

class TestExecuteCodeRuntimeError:

    def test_keyerror_on_bad_column(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "result = df['nonexistent_column'].sum()"
        res = execute_code(code, df_path)
        assert res.status == ExecutionStatus.RUNTIME_ERROR

    def test_syntax_error_in_user_code(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "result = df['score'.sum("  # broken syntax
        res = execute_code(code, df_path)
        assert res.status == ExecutionStatus.RUNTIME_ERROR

    def test_runtime_error_message_is_user_safe(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "result = 1 / 0"
        res = execute_code(code, df_path)
        assert res.status == ExecutionStatus.RUNTIME_ERROR
        # Temp directory paths must be sanitised from stderr and retry_instruction
        # (error_message is a hardcoded string and never contains paths)
        assert "runner.py" not in res.stderr
        assert "runner.py" not in res.retry_instruction

    def test_stderr_is_sanitised(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "result = df['bad_col']"
        res = execute_code(code, df_path)
        # runner.py must be replaced with the safe label regardless of OS temp dir format
        assert "runner.py" not in res.stderr
        assert "runner.py" not in res.retry_instruction

    def test_runtime_error_retry_instruction_includes_error(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "result = df['nonexistent'].mean()"
        res = execute_code(code, df_path)
        # retry_instruction should contain enough info for the LLM to self-correct
        assert len(res.retry_instruction) > 20


# ---------------------------------------------------------------------------
# execute_code — TIMEOUT
# ---------------------------------------------------------------------------

class TestExecuteCodeTimeout:

    def test_timeout_status(self, tmp_path):
        df_path = _simple_df(tmp_path)
        # Infinite loop — will always time out
        code = "while True: pass"
        res = execute_code(code, df_path, timeout_s=2)
        assert res.status == ExecutionStatus.TIMEOUT

    def test_timeout_error_message_mentions_duration(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "while True: pass"
        res = execute_code(code, df_path, timeout_s=2)
        assert "2" in res.error_message  # timeout duration referenced

    def test_timeout_retry_instruction_suggests_vectorisation(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "while True: pass"
        res = execute_code(code, df_path, timeout_s=2)
        # Should suggest pandas vectorisation rather than just "try again"
        assert any(
            word in res.retry_instruction.lower()
            for word in ["pandas", "vectori", "groupby", "efficient"]
        )

    def test_timeout_result_is_none(self, tmp_path):
        df_path = _simple_df(tmp_path)
        code = "while True: pass"
        res = execute_code(code, df_path, timeout_s=2)
        assert res.result is None


# ---------------------------------------------------------------------------
# ExecutionResult dataclass
# ---------------------------------------------------------------------------

class TestExecutionResult:

    def test_defaults_are_safe(self):
        res = ExecutionResult(status=ExecutionStatus.SUCCESS)
        assert res.result is None
        assert res.stdout == ""
        assert res.stderr == ""
        assert res.forbidden_module is None
        assert res.error_message == ""
        assert res.retry_instruction == ""

    def test_all_statuses_are_distinct(self):
        statuses = list(ExecutionStatus)
        assert len(statuses) == len(set(statuses))