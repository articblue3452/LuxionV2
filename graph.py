from langgraph.graph import StateGraph
from state import LuxionState
from nodes import classify_intent

graph = StateGraph(LuxionState)
graph.add_node("intent",classify_intent)
graph.set_entry_point("intent")
graph.set_finish_point("intent")
app = graph.compile()