from langgraph.graph import END, StateGraph
from state import LuxionState
from nodes import classify_intent, code_writer, executor, planner, reflection

graph = StateGraph(LuxionState)
graph.add_node("intent", classify_intent)
graph.add_node("planner", planner)
graph.add_node("code_writer", code_writer)
graph.add_node("executor", executor)
graph.add_node("reflection", reflection)

graph.set_entry_point("intent")
graph.add_edge("intent", "planner")
graph.add_edge("planner", "code_writer")
graph.add_edge("code_writer", "executor")
graph.add_edge("executor", "reflection")
graph.add_conditional_edges(
    "reflection",
    lambda state: "code_writer" if state["reflection"].get("retry") else END,
    {"code_writer": "code_writer", END: END},
)
app = graph.compile()
