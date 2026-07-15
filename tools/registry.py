from tools.base import Tool

from tools.file_tools import (
    read_file,
    run_python,
    write_file,
)

TOOLS = {

    "read_file": Tool(
        name="read_file",
        description="Read the contents of a text file.",
        function=read_file,
        parameters=["path"],
    ),

    "write_file": Tool(
        name="write_file",
        description="Write content into a text file.",
        function=write_file,
        parameters=["path", "content"],
    ),

    "run_python": Tool(
        name="run_python",
        description="Run a Python file and return stdout, stderr, and exit code.",
        function=run_python,
        parameters=["path"],
    ),
}
