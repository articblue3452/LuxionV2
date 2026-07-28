"""The extensibility boundary for Luxion capabilities.

Add a Tool here (or through a future plugin provider) and the planner receives
its description and input preconditions automatically.
"""

from tools.base import Tool
from tools.code_tools import code_writer
from tools.file_tools import read_file, run_python, write_file
from tools.response_tools import llm_answer
from tools.web_tools import web_search

TOOLS = {
    "read_file": Tool("read_file", "Read a UTF-8 text file.", read_file, ["path"]),
    "write_file": Tool("write_file", "Create or replace a UTF-8 text file.", write_file, ["path", "content"]),
    "run_python": Tool("run_python", "Run a Python file and return stdout, stderr, and exit code.", run_python, ["path"]),
    "web_search": Tool("web_search", "Search public web sources for current external information.", web_search, ["query"]),
    "llm_answer": Tool("llm_answer", "Produce the user-facing answer from the goal and prior tool results.", llm_answer, ["question"]),
    "code_writer": Tool("code_writer", "Generate complete content for one file; follow with write_file to persist it.", code_writer, ["path", "requirements"]),
    # Compatibility alias. New plans should choose llm_answer.
    "explain": Tool("explain", "Legacy alias for llm_answer.", llm_answer, ["question"]),
}


def planner_tool_catalog() -> list[dict[str, object]]:
    return [
        {"name": tool.name, "description": tool.description, "required_inputs": tool.parameters,
         "optional_inputs": tool.optional_parameters}
        for tool in TOOLS.values()
    ]
