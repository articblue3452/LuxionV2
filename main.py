from graph import app
result = app.invoke(
    {
        "user_input": "creat calculater with python",
        "intent": "",
        "plan":[]
    }
)
print(result)