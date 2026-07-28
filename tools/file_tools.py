"""Workspace-bounded file and Python execution tools."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

from tools.code_tools import validate_generated_content


def _workspace_path(path: str) -> Path:
    """Resolve a path and prevent tools from escaping the active workspace."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("path must be a non-empty string.")
    workspace = Path.cwd().resolve()
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (workspace / candidate).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise ValueError("Path must stay inside the Luxion workspace.") from error
    return resolved


def read_file(path: str) -> str:
    resolved = _workspace_path(path)
    return resolved.read_text(encoding="utf-8")


def write_file(path: str, content: str) -> str:
    resolved = _workspace_path(path)
    if not isinstance(content, str):
        raise ValueError("write_file content must be a string.")
    validation_error = validate_generated_content(str(resolved), content)
    if validation_error:
        raise ValueError(validation_error)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return "Success"


def run_python(path: str) -> dict[str, str | int | None]:
    resolved = _workspace_path(path)
    if resolved.suffix.lower() != ".py":
        raise ValueError("run_python requires a .py file.")
    try:
        result = subprocess.run(
            [sys.executable, str(resolved)],
            capture_output=True,
            input="",
            text=True,
            check=False,
            timeout=10,
            cwd=str(Path.cwd()),
        )
    except subprocess.TimeoutExpired as error:
        return {"returncode": None, "stdout": error.stdout or "", "stderr": "Process timed out after 10 seconds."}
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
