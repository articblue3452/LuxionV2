from tools.base import Tool

from tools.file_tools import (
    read_file,
    run_python,
    write_file,
)
from tools.response_tools import explain
from tools.web_tools import web_search

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

    "web_search": Tool(
        name="web_search",
        description="Search the public web for current, documented, or external information.",
        function=web_search,
        parameters=["query"],
    ),

    "explain": Tool(
        name="explain",
        description="Synthesize a direct answer, optionally using earlier web-search results.",
        function=explain,
        parameters=["question"],
    ),
}
