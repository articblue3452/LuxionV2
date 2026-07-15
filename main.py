from graph import app
result = app.invoke(
    {
        "user_input": "creat calculater with python",
        "intent": "",
        "plan":[],
        "last_result": None,
        "execution_results": [],
        "planner_error": None,
    }
)
print("\n========== FINAL STATE ==========\n")
print(result)
