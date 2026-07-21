from graph import app
result = app.invoke(
    {
        "user_input": "creat calculater with python in that user can give input only use python ",
        "semantic_memory": {},
        "memory_decision": {},
        "intent": "",
        "plan":[],
        "last_result": None,
        "execution_results": [],
        "planner_error": None,
        "retry_count": 0,
        "reflection": None,
    }
)
print("\n========== FINAL STATE ==========\n")
print(result)
