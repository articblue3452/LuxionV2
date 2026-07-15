import ast
import json
from typing import Any

from langchain_ollama import ChatOllama

from state import LuxionState
from tools.registry import TOOLS


def get_llm():
    return ChatOllama(model="hermes3", temperature=0)


def clean_json_content(content: str) -> str:
    content = content.strip()

    if content.startswith("```json"):
        content = content.replace("```json", "", 1).strip()

    if content.startswith("```"):
        content = content.replace("```", "", 1).strip()

    if content.endswith("```"):
        content = content[:-3].strip()

    start = content.find("{")
    end = content.rfind("}")

    if start != -1 and end != -1 and end > start:
        return content[start:end + 1]

    return content


def parse_json_response(content: str) -> dict[str, Any]:
    return json.loads(clean_json_content(content))


def clean_code_content(content: str) -> str:
    content = content.strip()

    if content.startswith("```python"):
        content = content.replace("```python", "", 1).strip()

    if content.startswith("```"):
        content = content.replace("```", "", 1).strip()

    if content.endswith("```"):
        content = content[:-3].strip()

    prose_markers = (
        "This code ",
        "The code ",
        "Explanation:",
        "Here is",
        "Here are",
    )

    lines = content.splitlines()
    cleaned_lines = []

    for line in lines:
        if line.startswith(prose_markers):
            break

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def validate_code_content(path: str, content: str, interactive_allowed: bool) -> str | None:
    if not content:
        return "Generated file content is empty."

    if path.endswith(".py"):
        try:
            ast.parse(content)
        except SyntaxError as e:
            return f"Python syntax error: {e}"

        if not interactive_allowed and "input(" in content:
            return "Generated code uses input(), but this task must run without interactive input."

    return None


def validate_plan(plan: Any) -> tuple[bool, str | None]:
    if not isinstance(plan, list):
        return False, "Plan must be a list."

    for index, step in enumerate(plan, start=1):
        if not isinstance(step, dict):
            return False, f"Step {index} must be an object."

        for key in ("step", "tool", "description", "args"):
            if key not in step:
                return False, f"Step {index} is missing '{key}'."

        tool_name = step["tool"]
        args = step["args"]

        if tool_name not in TOOLS:
            return False, f"Step {index} uses unknown tool '{tool_name}'."

        if not isinstance(args, dict):
            return False, f"Step {index} args must be an object."

        required_args = set(TOOLS[tool_name].parameters)
        provided_args = set(args)

        missing_args = required_args - provided_args
        extra_args = provided_args - required_args

        if missing_args:
            return False, f"Step {index} is missing args: {sorted(missing_args)}."

        if extra_args:
            return False, f"Step {index} has unknown args: {sorted(extra_args)}."

    return True, None


def normalize_plan(plan: list[dict]) -> list[dict]:
    normalized_plan = []
    seen_write_paths = set()

    for step in plan:
        step = dict(step)
        tool_name = step["tool"]
        args = step.get("args", {})

        if tool_name == "read_file":
            continue

        if tool_name == "write_file":
            if not isinstance(args, dict) or "path" not in args:
                normalized_plan.append(step)
                continue

            path = args["path"]

            if path in seen_write_paths:
                continue

            seen_write_paths.add(path)
            step["args"] = {
                "path": path,
                "content": "",
            }

        normalized_plan.append(step)

    for index, step in enumerate(normalized_plan, start=1):
        step["step"] = index

    return normalized_plan


def classify_intent(state: LuxionState):
    prompt = f"""
You are an Intent Classifier.

Classify the user's request into exactly one intent:
create
edit
run
test
optimize
explain

Rules:
- Return only one word.
- Do not explain.
- Do not use punctuation.

User Request:
{state["user_input"]}
"""

    response = get_llm().invoke(prompt)
    state["intent"] = response.content.strip().lower()
    return state


def planner(state: LuxionState):
    tool_text = ""

    for tool in TOOLS.values():
        tool_text += (
            f"Tool: {tool.name}\n"
            f"Description: {tool.description}\n"
            f"Parameters: {tool.parameters}\n\n"
        )

    prompt = f"""
You are Luxion Planner.

Your only job is to create an execution plan.
Do not write source code.
Do not solve the task.
Do not explain.

Available tools:
{tool_text}

Rules:
1. Return only valid JSON.
2. Return one JSON object with one key: plan.
3. plan must be a list.
4. Every step must contain exactly these keys: step, tool, description, args.
5. tool must be one of the available tools.
6. args keys must exactly match the selected tool parameters.
7. For write_file, set content to an empty string. The code_writer node writes the code later.
8. Use one write_file step per file.
9. For create tasks, use write_file first and run_python second.
10. Do not use read_file unless the user explicitly asks to inspect an existing file.
11. Do not say that run_python generates code. It only runs or tests a file.
12. Keep the plan short.

Output example:
{{
    "plan": [
        {{
            "step": 1,
            "tool": "write_file",
            "description": "Create calculator.py with calculator code",
            "args": {{
                "path": "calculator.py",
                "content": ""
            }}
        }},
        {{
            "step": 2,
            "tool": "run_python",
            "description": "Run calculator.py",
            "args": {{
                "path": "calculator.py"
            }}
        }}
    ]
}}

Intent:
{state["intent"]}

Goal:
{state["user_input"]}
"""

    response = get_llm().invoke(prompt)

    print("\n========== RAW PLANNER OUTPUT ==========")
    print(response.content)
    print("========================================\n")

    try:
        parsed = parse_json_response(response.content)
        plan = normalize_plan(parsed["plan"])
        is_valid, error = validate_plan(plan)

        if not is_valid:
            raise ValueError(error)

        state["plan"] = plan
        state["planner_error"] = None

    except Exception as e:
        state["plan"] = []
        state["planner_error"] = str(e)
        state["last_result"] = {
            "success": False,
            "error": f"Planner failed: {e}",
            "raw_output": response.content,
        }

    return state


def code_writer(state: LuxionState):
    if state.get("planner_error"):
        return state

    updated_plan = []
    interactive_allowed = any(
        word in state["user_input"].lower()
        for word in ("interactive", "input", "prompt", "ask user")
    )

    for step in state["plan"]:
        step = dict(step)

        if step["tool"] == "write_file":
            path = step["args"]["path"]
            error = None
            content = ""

            for attempt in range(1, 4):
                feedback = ""

                if error:
                    feedback = f"\nPrevious output was rejected because: {error}\nFix it now.\n"

                prompt = f"""
You are Luxion Code Writer.

Write the complete source code for the requested file.
Return only the file content.
Do not use markdown.
Do not wrap the code in backticks.
Do not explain.
The file must run without interactive input.
Do not use input() unless the user explicitly asks for an interactive program.
If useful, include a small non-interactive demo under if __name__ == "__main__".
{feedback}

User goal:
{state["user_input"]}

File path:
{path}

Step description:
{step["description"]}
"""

                response = get_llm().invoke(prompt)
                content = clean_code_content(response.content)
                error = validate_code_content(path, content, interactive_allowed)

                if not error:
                    break

            if error:
                state["last_result"] = {
                    "success": False,
                    "error": f"Code writer failed for {path}: {error}",
                    "content": content,
                }
                return state

            step["args"] = {
                "path": path,
                "content": content,
            }

        updated_plan.append(step)

    is_valid, error = validate_plan(updated_plan)

    if not is_valid:
        state["last_result"] = {
            "success": False,
            "error": f"Code writer produced invalid plan: {error}",
        }
        return state

    state["plan"] = updated_plan
    return state


def executor(state: LuxionState):
    if state.get("planner_error"):
        return state

    print("\n========== EXECUTOR ==========\n")

    results = []
    is_valid, error = validate_plan(state["plan"])

    if not is_valid:
        state["last_result"] = {
            "success": False,
            "error": error,
        }
        return state

    for step in state["plan"]:
        tool_name = step["tool"]
        args = step["args"]

        print(f"Executing Step {step['step']}")
        print(f"Tool : {tool_name}")
        print(f"Args : {args}")

        tool = TOOLS[tool_name]

        try:
            output = tool.function(**args)
            success = True

            if isinstance(output, dict) and "returncode" in output and output["returncode"] != 0:
                success = False

            result = {
                "success": success,
                "step": step["step"],
                "tool": tool_name,
                "output": output,
            }
            results.append(result)
            state["last_result"] = result

            if not success:
                state["execution_results"] = results
                return state

        except Exception as e:
            result = {
                "success": False,
                "step": step["step"],
                "tool": tool_name,
                "error": str(e),
            }
            results.append(result)
            state["execution_results"] = results
            state["last_result"] = result
            return state

    state["execution_results"] = results
    return state
