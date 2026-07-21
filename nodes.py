import ast
import json
import re
from typing import Any

from langchain_ollama import ChatOllama

from memory.semantic import SemanticMemory
from state import LuxionState
from tools.registry import TOOLS


def get_llm():
    return ChatOllama(model="hermes3", temperature=0)


def extract_semantic_preferences(user_input: str) -> dict[str, str]:
    """Extract only explicit, stable preferences from a user message."""
    preferences = {}

    language_match = re.search(
        r"\b(?:always\s+use|only\s+use|prefer(?:red)?\s+language\s+is|"
        r"i\s+prefer)\s+(python|javascript|typescript|java|c\+\+|c#|go|rust)\b",
        user_input,
        flags=re.IGNORECASE,
    )
    if language_match:
        language = language_match.group(1).lower()
        preferences["preferred_language"] = {
            "python": "Python",
            "javascript": "JavaScript",
            "typescript": "TypeScript",
            "java": "Java",
            "c++": "C++",
            "c#": "C#",
            "go": "Go",
            "rust": "Rust",
        }[language]

    framework_match = re.search(
        r"\b(?:always\s+use|only\s+use|prefer(?:red)?\s+framework\s+is|"
        r"my\s+favorite\s+framework\s+is)\s+(fastapi|django|flask)\b",
        user_input,
        flags=re.IGNORECASE,
    )
    if framework_match:
        framework = framework_match.group(1).lower()
        preferences["preferred_framework"] = {
            "fastapi": "FastAPI",
            "django": "Django",
            "flask": "Flask",
        }[framework]

    return preferences


def memory(state: LuxionState):
    """Save explicit preferences, then load them before intent and planning."""
    semantic_memory = SemanticMemory()
    preferences = extract_semantic_preferences(state["user_input"])

    for key, value in preferences.items():
        semantic_memory.set(key, value)

    state["semantic_memory"] = semantic_memory.all()
    return state


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
    memory_text = json.dumps(state.get("semantic_memory", {}), indent=2)

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
13. Apply known user preferences when they are relevant, unless the user
    explicitly asks for something different.

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

Known user preferences:
{memory_text}

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
    reflection_feedback = ""
    previous_reflection = state.get("reflection") or {}

    if previous_reflection.get("retry"):
        reflection_feedback = (
            "\nReflection feedback from the previous execution:\n"
            f"{previous_reflection.get('fix_instructions', '')}\n"
            "Correct the issue in this file while preserving the user's goal.\n"
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
{reflection_feedback}

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


def reflection(state: LuxionState):
    """Review execution output and request a bounded code repair when needed."""
    retry_count = state.get("retry_count", 0)

    if state.get("planner_error"):
        state["reflection"] = {
            "approved": False,
            "reason": "The planner failed, so there is no executable plan to review.",
            "fix_instructions": state["planner_error"],
            "retry": False,
        }
        return state

    execution_failed = any(
        not result.get("success", False)
        for result in state.get("execution_results", [])
    )

    interactive_run_skipped = any(
        isinstance(result.get("output"), dict)
        and result["output"].get("skipped") is True
        for result in state.get("execution_results", [])
    )

    if interactive_run_skipped and not execution_failed:
        state["reflection"] = {
            "approved": True,
            "reason": (
                "The interactive program passed syntax validation; its run was "
                "intentionally skipped because it requires user input."
            ),
            "fix_instructions": "",
            "retry": False,
        }
        return state

    execution_summary = json.dumps(state.get("execution_results", []), default=str)

    prompt = f"""
You are Luxion Reflection.

Review whether the execution fulfilled the user's goal. Identify the concrete
cause of any error and give concise, actionable instructions for the code
writer. Do not write source code.

A result whose output has "skipped": true is expected for a program that
uses input(). Its syntax was already validated, so do not reject it solely
because its interactive run was skipped.

Return only valid JSON with exactly these keys:
approved (boolean), reason (string), fix_instructions (string).

Example:
{{
  "approved": false,
  "reason": "The program failed because add() was not defined.",
  "fix_instructions": "Define add(a, b) before calling it, then run the file again."
}}

User goal:
{state["user_input"]}

Execution results:
{execution_summary}
"""

    try:
        reviewed = parse_json_response(get_llm().invoke(prompt).content)
        approved = reviewed.get("approved") is True and not execution_failed
        reason = str(reviewed.get("reason", "Reflection did not provide a reason."))
        fix_instructions = str(reviewed.get("fix_instructions", reason))
    except Exception as e:
        approved = not execution_failed
        reason = f"Reflection response could not be parsed: {e}"
        fix_instructions = (
            "Use the execution error above to correct the generated code and run it again."
        )

    should_retry = not approved and retry_count < 3
    if should_retry:
        state["retry_count"] = retry_count + 1

    state["reflection"] = {
        "approved": approved,
        "reason": reason,
        "fix_instructions": fix_instructions,
        "retry": should_retry,
    }
    return state


def is_interactive_python_run(step: dict, plan: list[dict]) -> bool:
    """Return True when this run_python step targets generated code using input()."""
    if step["tool"] != "run_python":
        return False

    target_path = step["args"].get("path")
    for plan_step in plan:
        if plan_step["tool"] != "write_file":
            continue

        write_args = plan_step["args"]
        if write_args.get("path") != target_path:
            continue

        try:
            tree = ast.parse(write_args.get("content", ""))
        except SyntaxError:
            return False

        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "input"
            for node in ast.walk(tree)
        )

    return False


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

        if is_interactive_python_run(step, state["plan"]):
            result = {
                "success": True,
                "step": step["step"],
                "tool": tool_name,
                "output": {
                    "skipped": True,
                    "reason": "Skipped run because this program requires interactive input.",
                },
            }
            results.append(result)
            state["last_result"] = result
            print("Skipped: program requires interactive input.")
            continue

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
