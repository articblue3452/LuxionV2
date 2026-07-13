from langchain_ollama import ChatOllama
from state import LuxionState
import json
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

    return state

def planner(state: LuxionState):

    prompt = f"""
You are Luxion Planner.

========================
ROLE
========================

You are responsible ONLY for planning.

You never execute code.
You never answer the user.
You never explain code.
You never solve the task.

Your only job is to convert the user's goal into a structured execution plan.

========================
TASK
========================

Analyze the user's intent and goal.

Break the goal into small logical steps.

Each step should represent ONE action only.

========================
RULES
========================

1. Return ONLY valid JSON.
2. Never return markdown.
3. Never wrap JSON inside ``` blocks.
4. Never explain your reasoning.
5. Never add extra text before or after JSON.
6. Keep the plan as short as possible.
7. Do not skip important steps.
8. Every step must contain:
   - step
   - action
   - description
9. Step numbers must start from 1.
10. Actions should describe WHAT should happen, not HOW it is implemented.

========================
OUTPUT FORMAT
========================

{{
    "plan": [
        {{
            "step": 1,
            "action": "Create a new Python file",
            "description": "Create calculator.py"
        }},
        {{
            "step": 2,
            "action": "Write program",
            "description": "Implement calculator logic"
        }}
    ]
}}

========================
FEW SHOT EXAMPLES
========================

Example 1

Intent:
create

Goal:
Create hello.py

Output

{{
    "plan": [
        {{
            "step":1,
            "action":"Create a new Python file",
            "description":"Create hello.py"
        }},
        {{
            "step":2,
            "action":"Write program",
            "description":"Write hello world program"
        }},
        {{
            "step":3,
            "action":"Run program",
            "description":"Execute hello.py"
        }}
    ]
}}

------------------------

Example 2

Intent:
optimize

Goal:
Optimize bubble sort

Output

{{
    "plan":[
        {{
            "step":1,
            "action":"Analyze existing implementation",
            "description":"Inspect bubble sort algorithm"
        }},
        {{
            "step":2,
            "action":"Improve algorithm",
            "description":"Replace inefficient logic"
        }},
        {{
            "step":3,
            "action":"Test solution",
            "description":"Verify optimized version"
        }}
    ]
}}

========================
CURRENT TASK
========================

Intent:
{state["intent"]}

Goal:
{state["user_input"]}

Return ONLY JSON.
"""

    response = llm.invoke(prompt)

    print("\n========== RAW PLANNER OUTPUT ==========")
    print(response.content)
    print("========================================\n")

    try:
        parsed = json.loads(response.content)
        state["plan"] = parsed["plan"]

    except Exception:

        state["plan"] = [
            {
                "step": 1,
                "action": "Planner Error",
                "description": "Planner returned invalid JSON."
            }
        ]

    return state
