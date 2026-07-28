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
    return state


def _validate_next_action(payload: dict[str, Any]) -> tuple[bool, str | None]:
    if payload.get("finished") is True:
        return (set(payload) == {"finished"}, "finished responses may contain only 'finished'.")
    if payload.get("finished") is not False or set(payload) != {"finished", "next_step"}:
        return False, "Planner response must contain finished and next_step only."
    step = payload["next_step"]
    if not isinstance(step, dict) or set(step) != {"tool", "args"} or not isinstance(step["tool"], str):
        return False, "next_step must contain only tool and args."
    tool = TOOLS.get(step["tool"])
    if tool is None:
        return False, f"Unknown tool: {step['tool']}."
    error = tool.validate_args(step["args"])
    return (error is None, error)


def planner(state: LuxionState) -> LuxionState:
    """The brain: return one valid action, then yield execution to the graph."""
    if state.get("finished"):
        return state
    if len(state.get("tool_history", [])) >= MAX_TOOL_STEPS:
        state.update(finished=True, response=state.get("response") or "Stopped after reaching the safety limit of 20 tool actions.")
        return state

    context = dict(state.get("context", {}))
    context["previous_execution_results"] = state.get("execution_history", [])
    context["last_result"] = state.get("last_result")
    context["reflection"] = state.get("reflection")
    prompt = f"""You are Luxion's Planner. Decide ONLY the next best action toward
the goal. You never execute a tool, write files, generate file contents, or
produce a user-facing answer yourself.

Return ONLY one JSON object in exactly one of these forms:
{{"finished": true}}
{{"finished": false, "next_step": {{"tool": "tool_name", "args": {{...}}}}}}

Rules:
- Select exactly one tool when unfinished.
- Respect every tool's required_inputs. Never call a tool until all of its inputs
  are already available; obtain missing content or facts with earlier actions.
- For code creation: code_writer returns content, then call write_file with that
  returned content. Do not write a file before content exists.
- Use web_search only for current or external knowledge.
- Use llm_answer as the final user-facing action when an explanation or summary
  is needed. After its result, return finished true.
- Inspect failed results and reflection before retrying; do not blindly repeat.

Goal: {json.dumps(state.get('goal', {}), default=str)}
Unified context: {json.dumps(context, default=str)}
Available tools: {json.dumps(planner_tool_catalog(), default=str)}"""
    try:
        decision = parse_json_response(get_llm().invoke(prompt).content)
        valid, error = _validate_next_action(decision)
        if not valid:
            raise ValueError(error)
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
            result = {"success": success, "tool": tool_name, "args": args, "output": output}
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
