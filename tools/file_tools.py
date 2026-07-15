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

    result = subprocess.run(
        [sys.executable, str(path)],
        capture_output=True,
        text=True,
        check=False,
    )

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
