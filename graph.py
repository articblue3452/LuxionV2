from langgraph.graph import StateGraph
from state import LuxionState
from nodes import classify_intent, planner

graph = StateGraph(LuxionState)
graph.add_node("intent", classify_intent)
graph.add_node("planner", planner)
graph.set_entry_point("intent")
graph.add_edge("intent", "planner")
graph.set_finish_point("planner")
app = graph.compile()