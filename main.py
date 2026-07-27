from graph import app
result = app.invoke(
    {
        "user_input": "scarch for todays news and store it in new file",
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
