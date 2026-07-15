from pathlib import Path
import subprocess
import sys


def read_file(path: str):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: str, content: str):
    path = Path(path)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    return "Success"


def run_python(path: str):
    path = Path(path)

    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            input="",
            text=True,
            check=False,
            timeout=10,
        )
    except subprocess.TimeoutExpired as e:
        return {
            "returncode": None,
            "stdout": e.stdout or "",
            "stderr": "Process timed out after 10 seconds.",
        }

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
