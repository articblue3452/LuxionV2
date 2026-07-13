from typing import TypedDict
class LuxionState(TypedDict):
    user_input:str
    intent:str
    plan: list[str]
