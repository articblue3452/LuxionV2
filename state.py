from typing import TypedDict
class LuxionState(TypedDict):
    user_input:str
    intent:str
    plan: list[dict]
    last_result: any
    