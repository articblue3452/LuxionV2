from langgraph.graph import StateGraph
from state import LuxionState
from nodes import classify_intent, code_writer, executor, planner

graph = StateGraph(LuxionState)
graph.add_node("intent", classify_intent)
graph.add_node("planner", planner)
graph.add_node("code_writer", code_writer)
graph.add_node("executor", executor)

graph.set_entry_point("intent")
graph.add_edge("intent", "planner")
graph.add_edge("planner", "code_writer")
graph.add_edge("code_writer", "executor")
graph.set_finish_point("executor")
app = graph.compile()
