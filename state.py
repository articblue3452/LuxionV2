"""State contract for Luxion's goal-driven agent loop.

Legacy fields remain available so integrations written for the original coding
agent do not fail while they migrate to the new fields.
"""

from typing import Any, TypedDict


class LuxionState(TypedDict, total=False):
    # Request and conversation
    user_input: str
    conversation: list[dict[str, Any]]
    context: dict[str, Any]
    goal: dict[str, Any]
    task_state: dict[str, Any]

    # Durable and runtime memory
    semantic_memory: dict[str, Any]
    memory_decision: dict[str, Any]
    execution_history: list[dict[str, Any]]
    tool_history: list[dict[str, Any]]
    last_tool: dict[str, Any] | None
    last_result: Any
    reflection: dict[str, Any] | None

    # Agent-loop control
    next_step: dict[str, Any] | None
    retry_count: int
    finished: bool
    response: str
    planner_error: str | None

    # Backward-compatible original coding-agent fields.
    intent: str
    route: str
    plan: list[dict[str, Any]]
    execution_results: list[dict[str, Any]]
