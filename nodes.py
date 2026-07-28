"""LangGraph nodes for Luxion's general-purpose, one-action agent loop."""

from __future__ import annotations

import json
from typing import Any

from langchain_ollama import ChatOllama

from memory.semantic import SemanticMemory
from state import LuxionState
from tools.registry import TOOLS, planner_tool_catalog

MAX_TOOL_STEPS = 20


def get_llm() -> ChatOllama:
    return ChatOllama(model="hermes3", temperature=0)


def clean_json_content(content: str) -> str:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:].strip()
    elif content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()
    start, end = content.find("{"), content.rfind("}")
    return content[start : end + 1] if start >= 0 and end > start else content


def parse_json_response(content: str) -> dict[str, Any]:
    parsed = json.loads(clean_json_content(content))
    if not isinstance(parsed, dict):
        raise ValueError("Model response must be a JSON object.")
    return parsed


def ignored_memory_decision(reason: str) -> dict[str, Any]:
    return {"remember": False, "reason": reason}


def validate_memory_decision(decision: Any) -> dict[str, Any]:
    if not isinstance(decision, dict) or not isinstance(decision.get("remember"), bool):
        raise ValueError("Memory decision requires boolean remember.")
    if not decision["remember"]:
        return ignored_memory_decision(str(decision.get("reason", "Not durable memory.")))
    key, value = decision.get("key"), decision.get("value")
    if decision.get("memory_type") != "semantic" or not isinstance(key, str) or not key.strip():
        raise ValueError("Semantic memory requires memory_type and key.")
    if value is None or isinstance(value, (dict, list)):
        raise ValueError("Semantic memory value must be scalar.")
    return {"remember": True, "memory_type": "semantic", "key": key.strip(), "value": value,
            "reason": str(decision.get("reason", "Durable preference."))}


def conversation_manager(state: LuxionState) -> LuxionState:
    """Normalize turn history without deciding what the request means."""
    conversation = list(state.get("conversation", []))
    user_input = state.get("user_input", "").strip()
    if not user_input:
        state.update(finished=True, response="Please provide a request.")
        return state
    if not conversation or conversation[-1] != {"role": "user", "content": user_input}:
        conversation.append({"role": "user", "content": user_input})
    state["conversation"] = conversation
    state.setdefault("tool_history", [])
    state.setdefault("execution_history", [])
    state.setdefault("execution_results", [])  # original public field
    state.setdefault("retry_count", 0)
    state.setdefault("finished", False)
    return state


def memory_retrieval(state: LuxionState) -> LuxionState:
    state["semantic_memory"] = SemanticMemory().all()
    return state


def context_builder(state: LuxionState) -> LuxionState:
    """Build the sole context object consumed by planning nodes."""
    state["context"] = {
        "user_request": state.get("user_input", ""),
        "conversation": state.get("conversation", []),
        "semantic_memory": state.get("semantic_memory", {}),
        "previous_execution_results": state.get("execution_history", []),
        "last_result": state.get("last_result"),
        "reflection": state.get("reflection"),
        "workspace_state": state.get("workspace_state", {}),  # extension point for workspace RAG
        "available_tools": planner_tool_catalog(),
    }
    return state


def goal_understanding(state: LuxionState) -> LuxionState:
    """Extract objective and constraints only; it never selects a tool."""
    prompt = f"""Extract the user's overall objective and constraints. Do not select
tools, make a plan, or answer the request. Return ONLY JSON:
{{"goal":"...","constraints":["..."]}}

User request: {state.get('user_input', '')}
Conversation: {json.dumps(state.get('conversation', []), default=str)}"""
    try:
        goal = parse_json_response(get_llm().invoke(prompt).content)
        if not isinstance(goal.get("goal"), str) or not goal["goal"].strip():
            raise ValueError("goal must be a non-empty string")
        constraints = goal.get("constraints", [])
        if not isinstance(constraints, list) or not all(isinstance(item, str) for item in constraints):
            raise ValueError("constraints must be strings")
        state["goal"] = {"goal": goal["goal"].strip(), "constraints": constraints}
    except Exception:
        # A usable fallback is preferable to blocking a normal user request.
        state["goal"] = {"goal": state.get("user_input", ""), "constraints": []}
    state.setdefault(
        "task_state",
        {
            "goal": state["goal"]["goal"],
            "completed_tasks": [],
            "remaining_tasks": ["Determine the next action needed to fulfil the goal."],
            "artifacts": {
                "search_results": [],
                "generated_content": None,
                "created_files": [],
                "execution_results": [],
            },
            "completed_actions": [],
            "goal_completed": False,
            "blocked": False,
            "block_reason": "",
        },
    )
    # Establish the first set of outstanding outcomes before the planner runs.
    requirements = _goal_requirements(state["goal"]["goal"])
    remaining: list[str] = []
    if requirements["search"]:
        remaining.append("Search for the required current information")
    if requirements["file"]:
        remaining.extend(["Generate content to save", "Write the requested file"])
    elif requirements["answer"]:
        remaining.append("Produce the user-facing answer")
    state["task_state"]["remaining_tasks"] = remaining
    return state


def _goal_requirements(goal: str) -> dict[str, bool]:
    """Infer only coarse completion requirements; artifacts remain the authority."""
    text = goal.lower()
    requires_search = any(word in text for word in ("search", "news", "today", "latest", "current"))
    requires_file = any(phrase in text for phrase in ("save", "store", "write", "file"))
    return {"search": requires_search, "file": requires_file, "answer": not requires_file}


def progress_evaluator(state: LuxionState) -> LuxionState:
    """Update durable task progress from artifacts; it never executes a tool.

    This deterministic boundary prevents a planner from declaring completion just
    because a tool call succeeded. It can be replaced by a richer evaluator
    later without changing planner or executor contracts.
    """
    task_state = state.setdefault("task_state", {})
    artifacts = task_state.setdefault("artifacts", {})
    artifacts.setdefault("search_results", [])
    artifacts.setdefault("generated_content", None)
    artifacts.setdefault("created_files", [])
    artifacts.setdefault("execution_results", [])
    task_state.setdefault("completed_actions", [])
    task_state.setdefault("completed_tasks", [])
    task_state.setdefault("blocked", False)
    task_state.setdefault("block_reason", "")

    result = state.get("last_result") or {}
    tool_name = result.get("tool")
    made_progress = False
    if result.get("success"):
        output = result.get("output")
        if tool_name == "web_search" and isinstance(output, list) and output:
            artifacts["search_results"] = output
            made_progress = True
        elif tool_name in {"llm_answer", "explain", "code_writer"} and isinstance(output, str) and output.strip():
            artifacts["generated_content"] = output
            made_progress = True
        elif tool_name == "write_file":
            path = result.get("args", {}).get("path")
            if isinstance(path, str) and path not in artifacts["created_files"]:
                artifacts["created_files"].append(path)
                made_progress = True
        elif tool_name == "run_python":
            artifacts["execution_results"].append(output)
            made_progress = True

        action = {"tool": tool_name, "args": result.get("args", {})}
        if action not in task_state["completed_actions"]:
            task_state["completed_actions"].append(action)

    if made_progress:
        task_state["no_progress_count"] = 0
    elif result:
        task_state["no_progress_count"] = task_state.get("no_progress_count", 0) + 1
        if task_state["no_progress_count"] >= 3:
            task_state["blocked"] = True
            task_state["block_reason"] = result.get("error", "Three actions made no task progress.")

    requirements = _goal_requirements(task_state.get("goal", state.get("goal", {}).get("goal", "")))
    remaining: list[str] = []
    if requirements["search"] and not artifacts["search_results"]:
        remaining.append("Search for the required current information")
    if requirements["file"] and not artifacts["generated_content"]:
        remaining.append("Generate content to save")
    if requirements["file"] and not artifacts["created_files"]:
        remaining.append("Write the requested file")
    if requirements["answer"] and not artifacts["generated_content"]:
        remaining.append("Produce the user-facing answer")

    completed: list[str] = []
    if artifacts["search_results"]:
        completed.append("searched web")
    if artifacts["generated_content"]:
        completed.append("generated content")
    if artifacts["created_files"]:
        completed.append("wrote requested file")
    if artifacts["execution_results"]:
        completed.append("ran requested Python program")

    task_state["completed_tasks"] = completed
    task_state["remaining_tasks"] = remaining
    task_state["goal_completed"] = not remaining
    state["task_state"] = task_state
    return state


def normalize_planner_decision(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep the meaningful planner fields while ignoring harmless LLM metadata.

    The model may add a ``reason`` or a step ``description`` despite being told
    not to. Those fields must not prevent a safe, fully validated action.
    """
    finished = payload.get("finished")
    if not isinstance(finished, bool):
        raise ValueError("Planner response requires boolean 'finished'.")
    if finished:
        return {"finished": True}
    step = payload.get("next_step")
    if not isinstance(step, dict):
        raise ValueError("Planner response requires object 'next_step' when unfinished.")
    return {
        "finished": False,
        "next_step": {"tool": step.get("tool"), "args": step.get("args")},
    }


def _validate_next_action(payload: dict[str, Any]) -> tuple[bool, str | None]:
    if payload.get("finished") is True:
        return True, None
    if payload.get("finished") is not False:
        return False, "Planner response requires boolean 'finished'."
    step = payload.get("next_step")
    if not isinstance(step, dict) or not isinstance(step.get("tool"), str):
        return False, "next_step requires a string tool name."
    tool = TOOLS.get(step["tool"])
    if tool is None:
        return False, f"Unknown tool: {step['tool']}."
    error = tool.validate_args(step["args"])
    return (error is None, error)


def planner(state: LuxionState) -> LuxionState:
    """The brain: return one valid action, then yield execution to the graph."""
    if state.get("finished"):
        return state
    if state.get("task_state", {}).get("blocked"):
        state.update(
            finished=True,
            next_step=None,
            response=("I could not complete the request because progress stopped: "
                      f"{state['task_state'].get('block_reason', 'unknown error')}"),
        )
        return state
    if state.get("task_state", {}).get("goal_completed"):
        state.update(finished=True, next_step=None)
        if not state.get("response"):
            state["response"] = "Completed the requested work."
        return state
    if len(state.get("tool_history", [])) >= MAX_TOOL_STEPS:
        state.update(finished=True, response=state.get("response") or "Stopped after reaching the safety limit of 20 tool actions.")
        return state

    prompt = f"""You are Luxion's Planner. Decide ONLY the next best action toward
the goal. You never execute a tool, write files, generate file contents, or
produce a user-facing answer yourself.

Return ONLY one JSON object in exactly one of these forms:
{{"finished": true}}
{{"finished": false, "next_step": {{"tool": "tool_name", "args": {{...}}}}}}

Rules:
- Select exactly one tool when unfinished.
- Start with task_state.remaining_tasks. Do not reconstruct task progress from
  raw execution history.
- Never repeat an item in task_state.completed_actions unless Reflection has
  explicitly requested a retry.
- Respect every tool's required_inputs. Never call a tool until all of its inputs
  are already available; obtain missing content or facts with earlier actions.
- For code creation: code_writer returns content, then call write_file with that
  returned content. Do not write a file before content exists.
- Use web_search only for current or external knowledge.
- Use llm_answer to synthesize an explanation or content from prior results.
  It can be followed by write_file if the goal requires saving that content.
- Inspect failed results and reflection before retrying; do not blindly repeat.

Artifact references: when saving generated content, use this exact value for
write_file.content: {{"artifact": "generated_content"}}. Never use a prose
placeholder such as "<the last result>".

Examples (copy the JSON shape exactly; never add a reason, description, prose,
or Markdown):
User goal: "Search today's AI news and save it."
First action:
{{"finished": false, "next_step": {{"tool": "web_search", "args": {{"query": "today's AI news"}}}}}}
After search, synthesize saveable content:
{{"finished": false, "next_step": {{"tool": "llm_answer", "args": {{"question": "Summarize the AI news results for a text file."}}}}}}
After llm_answer has supplied the content:
{{"finished": false, "next_step": {{"tool": "write_file", "args": {{"path": "ai_news.txt", "content": {{"artifact": "generated_content"}}}}}}}}
When every requested outcome is complete:
{{"finished": true}}

Goal: {json.dumps(state.get('goal', {}), default=str)}
Task state: {json.dumps(state.get('task_state', {}), default=str)}
Semantic memory: {json.dumps(state.get('semantic_memory', {}), default=str)}
Conversation: {json.dumps(state.get('conversation', []), default=str)}
Reflection: {json.dumps(state.get('reflection'), default=str)}
Available tools: {json.dumps(planner_tool_catalog(), default=str)}"""
    try:
        decision: dict[str, Any] | None = None
        validation_error = ""
        raw_output = ""
        for attempt in range(2):
            if attempt == 0:
                raw_output = get_llm().invoke(prompt).content
            else:
                repair_prompt = f"""Your previous planner output failed validation.
Validation error: {validation_error}
Previous output: {raw_output}

Return corrected JSON only. Use one of these exact schemas:
{{"finished": true}}
{{"finished": false, "next_step": {{"tool": "tool_name", "args": {{}}}}}}
Do not include Markdown, explanations, reason, or description."""
                raw_output = get_llm().invoke(repair_prompt).content
            try:
                candidate = normalize_planner_decision(parse_json_response(raw_output))
                if candidate["finished"] and not state.get("task_state", {}).get("goal_completed"):
                    raise ValueError(
                        "Progress Evaluator reports unfinished tasks: "
                        f"{state.get('task_state', {}).get('remaining_tasks', [])}. "
                        "Choose one action; Planner cannot declare completion."
                    )
                valid, validation_error = _validate_next_action(candidate)
                if valid:
                    decision = candidate
                    break
            except Exception as error:
                validation_error = str(error)
        if decision is None:
            raise ValueError(validation_error or "Planner returned no valid action.")
        state["planner_error"] = None
        state["finished"] = decision["finished"]
        state["next_step"] = None if decision["finished"] else decision["next_step"]
        if decision["finished"] and not state.get("response"):
            state["response"] = "Completed the requested work."
    except Exception as error:
        state.update(finished=True, planner_error=str(error), response=f"I could not form a safe next action: {error}")
    return state


def tool_executor(state: LuxionState) -> LuxionState:
    """Execute exactly the one prevalidated planner action."""
    step = state.get("next_step")
    if state.get("finished") or not step:
        return state
    tool_name, args = step["tool"], step["args"]
    tool = TOOLS[tool_name]
    error = tool.validate_args(args)
    completed_actions = state.get("task_state", {}).get("completed_actions", [])
    duplicate = {"tool": tool_name, "args": args} in completed_actions
    retry_requested = bool((state.get("reflection") or {}).get("retry"))
    if not error and duplicate and not retry_requested:
        error = "This exact action already completed; choose an unfinished task."
    if not error and tool_name == "write_file":
        content = args.get("content")
        if content == "<the last web_search result already in context>" or content == "<the last result>":
            error = "Placeholder file content is invalid; use a real artifact reference."
        elif isinstance(content, dict) and content.get("artifact") == "generated_content":
            generated = state.get("task_state", {}).get("artifacts", {}).get("generated_content")
            if not isinstance(generated, str) or not generated.strip():
                error = "generated_content artifact is not available yet."
            else:
                args = {**args, "content": generated}
        elif not isinstance(content, str) or not content.strip():
            error = "write_file content must be a non-empty string or generated_content artifact reference."
        elif content != state.get("task_state", {}).get("artifacts", {}).get("generated_content"):
            error = "write_file must use generated_content rather than planner-invented content."
    if error:
        result: dict[str, Any] = {"success": False, "tool": tool_name, "args": args, "error": error}
    else:
        try:
            if tool_name in {"llm_answer", "explain"}:
                output = tool.function(args["question"], research=json.dumps(state.get("execution_history", []), default=str))
                state["response"] = output
            else:
                output = tool.function(**args)
            success = not (isinstance(output, dict) and output.get("returncode") not in (None, 0))
            if tool_name == "web_search" and not output:
                success = False
                result_error = "Web search returned no results. Try a different available source or report that no current results were found."
            result = {"success": success, "tool": tool_name, "args": args, "output": output}
            if not success and tool_name == "web_search" and not output:
                result["error"] = result_error
        except Exception as exc:
            result = {"success": False, "tool": tool_name, "args": args, "error": str(exc)}
    state["last_tool"] = step
    state["last_result"] = result
    state.setdefault("tool_history", []).append({"tool": tool_name, "args": args})
    state.setdefault("execution_history", []).append(result)
    state["execution_results"] = state["execution_history"]  # compatibility
    state["next_step"] = None
    return state


def reflection(state: LuxionState) -> LuxionState:
    """Validate one execution result and supply retry guidance; never executes."""
    result = state.get("last_result") or {}
    if result.get("success"):
        state["reflection"] = {"approved": True, "reason": "The last tool completed successfully.", "fix_instructions": "", "retry": False}
    else:
        retry_count = state.get("retry_count", 0) + 1
        state["retry_count"] = retry_count
        state["reflection"] = {
            "approved": False,
            "reason": result.get("error", "Tool execution failed."),
            "fix_instructions": "Use the failure details to choose a corrected next action or ask the user if required information is unavailable.",
            "retry": retry_count <= 3,
        }
    return state


def memory_manager(state: LuxionState) -> LuxionState:
    """Keep the legacy semantic-memory policy, invoked after task completion."""
    prompt = f"""Decide whether this message contains a durable user preference.
Return only JSON. Remember only stable preferences or identity, never one-time
tasks. Forms: {{"remember":false,"reason":"..."}} or
{{"remember":true,"memory_type":"semantic","key":"...","value":"...","reason":"..."}}
Message: {state.get('user_input', '')}"""
    try:
        state["memory_decision"] = validate_memory_decision(parse_json_response(get_llm().invoke(prompt).content))
    except Exception as error:
        state["memory_decision"] = ignored_memory_decision(f"Memory decision unavailable: {error}")
    return state


def save_memory(state: LuxionState) -> LuxionState:
    decision = state.get("memory_decision", {})
    if decision.get("remember") and decision.get("memory_type") == "semantic":
        SemanticMemory().set(decision["key"], decision["value"])
    return state


def route_after_planner(state: LuxionState) -> str:
    return "memory_manager" if state.get("finished") else "tool_executor"


def route_after_reflection(state: LuxionState) -> str:
    return "planner"


def route_memory_completion(state: LuxionState) -> str:
    return "save_memory" if state.get("memory_decision", {}).get("remember") else "end"


# Legacy imports are deliberately retained. They are no longer graph nodes.
def classify_intent(state: LuxionState) -> LuxionState:
    state["intent"] = "goal_driven"
    return state


def router(state: LuxionState) -> LuxionState:
    state["route"] = "planner"
    return state


def route_intent(state: LuxionState) -> str:
    return "planner"


def explain(state: LuxionState) -> LuxionState:
    state["response"] = TOOLS["llm_answer"].function(state.get("user_input", ""))
    return state


def code_writer(state: LuxionState) -> LuxionState:
    """Compatibility no-op: generation now belongs to the code_writer tool."""
    return state


executor = tool_executor
