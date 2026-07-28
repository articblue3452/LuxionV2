from graph import app
result = app.invoke(
    {
        "user_input": "creat a file and save details of letest update of python",
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
