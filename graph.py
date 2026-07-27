from langgraph.graph import END, StateGraph
from state import LuxionState
from nodes import (
    classify_intent,
    code_writer,
    executor,
    explain,
    memory_manager,
    memory_retrieval,
    planner,
    reflection,
    route_intent,
    route_memory_completion,
    router,
    save_memory,
)

graph = StateGraph(LuxionState)
graph.add_node("memory_manager", memory_manager)
graph.add_node("save_memory", save_memory)
graph.add_node("memory_retrieval", memory_retrieval)
graph.add_node("intent", classify_intent)
graph.add_node("router", router)
graph.add_node("explain", explain)
graph.add_node("planner", planner)
graph.add_node("code_writer", code_writer)
graph.add_node("executor", executor)
graph.add_node("reflection", reflection)

graph.set_entry_point("memory_retrieval")
graph.add_edge("memory_retrieval", "intent")
graph.add_edge("intent", "router")
graph.add_conditional_edges(
    "router",
    route_intent,
    {"explain": "explain", "planner": "planner"},
)
graph.add_edge("explain", "memory_manager")
graph.add_edge("planner", "code_writer")
graph.add_edge("code_writer", "executor")
graph.add_edge("executor", "reflection")
graph.add_conditional_edges(
    "reflection",
    lambda state: "code_writer" if state["reflection"].get("retry") else "memory_manager",
    {"code_writer": "code_writer", "memory_manager": "memory_manager"},
)
graph.add_conditional_edges(
    "memory_manager",
    route_memory_completion,
    {"save_memory": "save_memory", "end": END},
)
graph.add_edge("save_memory", END)
app = graph.compile()
