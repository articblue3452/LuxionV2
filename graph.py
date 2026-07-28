"""Luxion's general-purpose agent graph."""

from langgraph.graph import END, StateGraph

from nodes import (
    context_builder,
    conversation_manager,
    goal_understanding,
    memory_manager,
    memory_retrieval,
    planner,
    reflection,
    route_after_planner,
    route_after_reflection,
    route_memory_completion,
    save_memory,
    tool_executor,
)
from state import LuxionState

graph = StateGraph(LuxionState)
graph.add_node("conversation_manager", conversation_manager)
graph.add_node("memory_retrieval", memory_retrieval)
graph.add_node("context_builder", context_builder)
graph.add_node("goal_understanding", goal_understanding)
graph.add_node("planner", planner)
graph.add_node("tool_executor", tool_executor)
graph.add_node("reflection", reflection)
graph.add_node("memory_manager", memory_manager)
graph.add_node("save_memory", save_memory)

graph.set_entry_point("conversation_manager")
graph.add_edge("conversation_manager", "memory_retrieval")
graph.add_edge("memory_retrieval", "context_builder")
graph.add_edge("context_builder", "goal_understanding")
graph.add_edge("goal_understanding", "planner")
graph.add_conditional_edges("planner", route_after_planner, {"tool_executor": "tool_executor", "memory_manager": "memory_manager"})
graph.add_edge("tool_executor", "reflection")
graph.add_conditional_edges("reflection", route_after_reflection, {"planner": "planner"})
graph.add_conditional_edges("memory_manager", route_memory_completion, {"save_memory": "save_memory", "end": END})
graph.add_edge("save_memory", END)

app = graph.compile()
