from langchain_ollama import ChatOllama
from state import LuxionState
from tools.registry import TOOLS
import json

llm = ChatOllama(model="hermes3")
# ==========================
# INTENT CLASSIFIER
# ==========================
def classify_intent(state: LuxionState):

    prompt = f"""
You are an Intent Classifier.

Your job is to classify the user's request.

Possible intents:

create
edit
run
test
optimize
explain

Rules:

- Return ONLY one word.
- Never explain.
- Never use punctuation.
- Never invent a new intent.

User Request:

{state["user_input"]}
"""


    response = llm.invoke(prompt)

    state["intent"] = response.content.strip().lower()

    return state


# ==========================
# PLANNER
# ==========================
def planner(state: LuxionState):

    # Build tool list dynamically
    tool_text = ""

    for tool in TOOLS.values():
        tool_text += (
            f"Tool: {tool.name}\n"
            f"Description: {tool.description}\n\n"
        )

    prompt = f"""
You are Luxion Planner.

=========================
ROLE
=========================

You ONLY create execution plans.

You NEVER execute code.

You NEVER answer the user.

You NEVER explain.

=========================
AVAILABLE TOOLS
=========================

{tool_text}

=========================
RULES
=========================

1. Return ONLY valid JSON.

2. Never use markdown.

3. Never wrap JSON in ```.

4. Never explain anything.

5. Use ONLY the tools listed above.

6. Every step MUST contain:

- step
- tool
- description
- args

7. args MUST always be a JSON object.

8. Never invent tool names.

9. One step = one action.

10. Keep the plan as short as possible.

=========================
OUTPUT FORMAT
=========================

{{
    "plan":[
        {{
            "step":1,
            "tool":"write_file",
            "description":"Create hello.py",
            "args":{{
                "path":"hello.py",
                "content":""
            }}
        }},
        {{
            "step":2,
            "tool":"write_file",
            "description":"Write hello world code",
            "args":{{
                "path":"hello.py",
                "content":"print('Hello World')"
            }}
        }},
        {{
            "step":3,
            "tool":"run_python",
            "description":"Run hello.py",
            "args":{{
                "path":"hello.py"
            }}
        }}
    ]
}}

=========================
CURRENT TASK
=========================

Intent:

{state["intent"]}

Goal:

{state["user_input"]}

Return ONLY valid JSON.
"""


    response = llm.invoke(prompt)

    print("\n========== RAW PLANNER OUTPUT ==========")
    print(response.content)
    print("========================================\n")

    try:

        content = response.content.strip()

        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()

        elif content.startswith("```"):
            content = content.replace("```", "").strip()

        parsed = json.loads(content)

        state["plan"] = parsed["plan"]

    except Exception as e:

        print("Planner JSON Error:", e)

        state["plan"] = [
            {
                "step": 1,
                "tool": "unknown",
                "description": "Planner failed to generate valid JSON.",
                "args": {}
            }
        ]

    return state
