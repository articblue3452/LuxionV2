from langgraph.graph import StateGraph
from state import LuxionState
from nodes import classify_intent, planner, executor

graph = StateGraph(LuxionState)
graph.add_node("intent", classify_intent)
graph.add_node("planner", planner)
graph.add_node("executor", executor)

graph.set_entry_point("intent")
graph.add_edge("intent", "planner")
graph.add_edge("planner", "executor")
graph.set_finish_point("executor")
app = graph.compile()