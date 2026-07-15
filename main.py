from graph import app
result = app.invoke(
    {
        "user_input": "creat calculater with python",
        "intent": "",
        "plan":[],
        "last_result":None
    }
)
print("\n========== FINAL STATE ==========\n")
print(result)