from langchain_ollama import ChatOllama
from state import LuxionState
llm = ChatOllama(model="hermes3")
def classify_intent(state: LuxionState):
    prompt = f"""
    You are an intent classifier.

You MUST classify the user's request into EXACTLY ONE of these intents:

create
edit
run
test
optimize
explain

Rules:
- Your answer MUST be one of the six words above.
- Never invent another word.
- Never explain.
- Never answer with anything except one of those six words.



    {state["user_input"]}
    """
    response = llm.invoke(prompt)
    state["intent"] = response.content.strip().lower()
    print(state)
    return state
