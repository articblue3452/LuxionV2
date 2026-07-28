"""Validated code generation used by the ``code_writer`` tool."""

from __future__ import annotations

import ast
from pathlib import Path

from langchain_ollama import ChatOllama


def clean_generated_content(content: str) -> str:
    """Remove a single Markdown fence without damaging valid source code."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        cleaned = cleaned[first_newline + 1 :] if first_newline >= 0 else ""
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def validate_generated_content(path: str, content: str) -> str | None:
    """Return a user-safe validation error before generated source is persisted."""
    if not isinstance(content, str) or not content.strip():
        return "Generated file content is empty."
    if content.lstrip().startswith("```") or content.rstrip().endswith("```"):
        return "Generated file content contains Markdown fences."
    if Path(path).suffix.lower() == ".py":
        try:
            tree = ast.parse(content)
        except SyntaxError as error:
            return f"Generated Python has invalid syntax: {error.msg} at line {error.lineno}."
        uses_input = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "input"
            for node in ast.walk(tree)
        )
        has_main_guard = any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            and any(isinstance(comparator, ast.Constant) and comparator.value == "__main__"
                    for comparator in node.test.comparators)
            for node in ast.walk(tree)
        )
        if uses_input and not has_main_guard:
            return "Interactive Python must include an if __name__ == '__main__' entry point that runs the program."
    return None


def code_writer(path: str, requirements: str) -> str:
    """Generate validated source content; persistence remains a separate tool."""
    if not isinstance(path, str) or not path.strip() or not isinstance(requirements, str) or not requirements.strip():
        raise ValueError("code_writer requires non-empty path and requirements.")

    previous_error = ""
    for _ in range(3):
        prompt = f"""You are Luxion's code generation tool. Generate the complete
contents for {path}. Return source code only: no Markdown fences, no prose, and
no explanation. The code must be syntactically valid. If it asks for user input
or a runnable program, include an if __name__ == "__main__" entry point that
calls the program; do not only define a function.

Requirements:
{requirements}
{previous_error}"""
        raw_content = ChatOllama(model="hermes3", temperature=0).invoke(prompt).content
        content = clean_generated_content(raw_content)
        error = validate_generated_content(path, content)
        if error is None:
            return content
        previous_error = f"\nYour previous output was rejected: {error} Return corrected source only."

    raise ValueError(f"code_writer could not generate valid content for {path}: {error}")
