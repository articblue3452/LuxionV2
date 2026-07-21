from typing import Any, TypedDict


class LuxionState(TypedDict):
    user_input: str
    semantic_memory: dict[str, Any]
    intent: str
    plan: list[dict]
    last_result: Any
    execution_results: list[dict]
    planner_error: str | None
    retry_count: int
    reflection: dict | None
