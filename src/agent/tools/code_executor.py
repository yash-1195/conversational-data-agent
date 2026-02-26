# src/agent/tools/code_executor.py

import ast
import pickle
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ExecutionStatus(Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    FORBIDDEN_IMPORT = "forbidden_import"
    RUNTIME_ERROR = "runtime_error"
    MISSING_RESULT = "missing_result"


ALLOWED_MODULES = frozenset({
    "pandas", "numpy", "matplotlib", "plotly",
    "math", "statistics", "datetime", "collections",
    "itertools", "functools", "re", "json", "string",
})

_DANGEROUS_CALLS = frozenset({"__import__", "exec", "eval", "compile"})

_MAX_RESULT_BYTES = 50 * 1024 * 1024  # 50 MB


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    result: Any = None               # the value of `result` variable if successful
    stdout: str = ""
    stderr: str = ""
    forbidden_module: str | None = None   # populated if FORBIDDEN_IMPORT
    error_message: str = ""              # human-readable, safe to surface to user
    retry_instruction: str = ""          # appended to prompt on retry


def _check_forbidden_imports(code: str) -> str | None:
    """
    Parse the code as AST.
    Rejects any import whose root module is not in ALLOWED_MODULES, and
    any call to __import__, exec, eval, or compile (dynamic import bypass).
    Returns a human-readable violation string, or None if clean.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # syntax errors are caught at runtime

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    return alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if root not in ALLOWED_MODULES:
                return module
        elif isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in _DANGEROUS_CALLS:
                return f"{name}() call"

    return None


def execute_code(code: str, dataframe_pickle_path: str, timeout_s: int = 30) -> ExecutionResult:
    """
    Execute `code` in a sandboxed subprocess. The code receives the user's
    DataFrame pre-loaded as `df`. It must assign its final output to `result`.

    Steps:
    1. AST check for forbidden imports (fast, no subprocess spawned)
    2. Write a runner script to a temp file
    3. Run in subprocess with timeout
    4. Deserialise `result` from a temp output file
    """

    # Step 1: static import check
    forbidden = _check_forbidden_imports(code)
    if forbidden:
        return ExecutionResult(
            status=ExecutionStatus.FORBIDDEN_IMPORT,
            forbidden_module=forbidden,
            error_message=f"Import of '{forbidden}' is not permitted in the sandbox.",
            retry_instruction=(
                f"Your code attempted '{forbidden}', which is not permitted in the sandbox. "
                f"Only these imports are allowed: {', '.join(sorted(ALLOWED_MODULES))}. "
                "Rewrite the code using only those libraries."
            )
        )

    # Step 2: build the runner script.
    # Built by list-join rather than str.format() so that { } characters
    # in user code (f-strings, dicts, sets) cannot corrupt the script.
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        out_path = tmp / "result.pkl"
        script_path = tmp / "runner.py"

        runner_lines = [
            "import pickle",
            "import sys",
            "import pandas as pd",
            "import numpy as np",
            "",
            f"with open({repr(str(dataframe_pickle_path))}, 'rb') as _f:",
            "    df = pickle.load(_f)",
            "",
            "# --- user code start ---",
            code.strip(),
            "# --- user code end ---",
            "",
            "try:",
            "    _result_bytes = pickle.dumps(result)",
            "except NameError:",
            "    sys.exit(42)",
            "",
            f"if len(_result_bytes) > {_MAX_RESULT_BYTES}:",
            "    print('Result too large. Reduce or summarise the output.', file=sys.stderr)",
            "    sys.exit(1)",
            "",
            f"with open({repr(str(out_path))}, 'wb') as _f:",
            "    _f.write(_result_bytes)",
        ]
        runner_script = "\n".join(runner_lines)
        script_path.write_text(runner_script, encoding="utf-8")

        # Step 3: run subprocess
        try:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error_message=f"Code execution timed out after {timeout_s} seconds.",
                retry_instruction=(
                    f"Your code exceeded the {timeout_s}s timeout. This usually means "
                    "an inefficient operation on a large DataFrame (e.g. a Python-level "
                    "loop instead of vectorised pandas). Rewrite using pandas built-ins "
                    "(groupby, apply, vectorised operations) to reduce execution time."
                )
            )

        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()

        # Exit code 42 = result variable was never defined
        if proc.returncode == 42:
            return ExecutionResult(
                status=ExecutionStatus.MISSING_RESULT,
                stdout=stdout,
                stderr=stderr,
                error_message="The code did not assign a value to 'result'.",
                retry_instruction=(
                    "Your code ran without errors but never assigned a value to 'result'. "
                    "The final output must always be assigned to a variable named exactly "
                    "`result`. For example: `result = df.groupby('category').size()`."
                )
            )

        # Non-zero exit = runtime error
        if proc.returncode != 0:
            # Sanitise stderr — remove file paths that expose temp dirs
            safe_stderr = _sanitise_traceback(stderr)
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                stdout=stdout,
                stderr=safe_stderr,
                error_message="Code raised an error during execution.",
                retry_instruction=(
                    f"Your code raised the following error:\n{safe_stderr}\n"
                    "Fix the error and try again. If a column name is wrong, check "
                    "the dataset schema provided above."
                )
            )

        # Step 4: deserialise result
        try:
            with open(out_path, "rb") as f:
                result_value = pickle.load(f)
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.RUNTIME_ERROR,
                error_message="Failed to read execution output.",
                retry_instruction="An internal error occurred reading the output. Try simplifying the result value."
            )

        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            result=result_value,
            stdout=stdout,
            stderr=stderr,
            error_message="",
            retry_instruction=""
        )


def _sanitise_traceback(stderr: str) -> str:
    """Remove temp directory paths from tracebacks before surfacing to user."""
    # Replace paths like /tmp/tmpXXXXX/runner.py with runner.py
    cleaned = re.sub(r'File ".*?runner\.py"', 'File "generated_code.py"', stderr)
    return cleaned