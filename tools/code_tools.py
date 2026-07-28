"""Code generation is a tool, not a graph node."""

from langchain_ollama import ChatOllama


def code_writer(path: str, requirements: str) -> str:
    """Generate complete file content. The planner must write it explicitly."""
    if not path.strip() or not requirements.strip():
        raise ValueError("code_writer requires path and requirements.")
    prompt = f"""You are Luxion's code generation tool. Generate the complete
contents for {path}. Return only file contents, with no Markdown or explanation.
Requirements:\n{requirements}"""
    return ChatOllama(model="hermes3", temperature=0).invoke(prompt).content.strip()
