# src/agent/tools/code_executor.py

import ast
import pickle
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ExecutionStatus(Enum):
    SUCCESS = "success"
    TIMEOUT = "timeout"
    FORBIDDEN_IMPORT = "forbidden_import"
    RUNTIME_ERROR = "runtime_error"
    MISSING_RESULT = "missing_result"


FORBIDDEN_MODULES = {"os", "sys", "subprocess", "socket", "shutil", "importlib",
                     "builtins", "ctypes", "multiprocessing", "threading", "signal",
                     "pty", "atexit", "gc"}


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
    Parse the code as AST and return the first forbidden module name found,
    or None if clean. Catches both `import os` and `from os import path`.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None  # syntax errors are caught at runtime, not here

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_MODULES:
                    return alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".")[0]
            if root in FORBIDDEN_MODULES:
                return module

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
                f"Your code attempted to import '{forbidden}', which is forbidden. "
                "Do not import os, sys, subprocess, socket, or any stdlib module "
                "outside of pandas, numpy, matplotlib, and plotly. Rewrite the code "
                "without that import."
            )
        )

    # Step 2: build the runner script
    # The runner loads df from the pickle path, executes user code,
    # then pickles `result` to a known output path.
    runner_template = textwrap.dedent("""
        import pickle, traceback, sys
        import pandas as pd
        import numpy as np

        # Load the dataframe
        with open({df_path!r}, "rb") as f:
            df = pickle.load(f)

        # --- user code start ---
        {user_code}
        # --- user code end ---

        # Serialise result
        try:
            with open({out_path!r}, "wb") as f:
                pickle.dump(result, f)
        except NameError:
            sys.exit(2)  # exit code 2 = result not defined
    """)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        out_path = tmp / "result.pkl"
        script_path = tmp / "runner.py"

        runner_script = runner_template.format(
            df_path=str(dataframe_pickle_path),
            user_code=code.strip(),
            out_path=str(out_path),
        )

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

        # Exit code 2 = result variable was never defined
        if proc.returncode == 2:
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
                error_message=f"Code raised an error during execution.",
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