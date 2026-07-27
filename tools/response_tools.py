"""Response tools for task plans that include research."""

from langchain_ollama import ChatOllama


def explain(question: str, research: str = "") -> str:
    """Produce a concise answer from a question and optional research context."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("explain requires a non-empty question.")

    prompt = f"""
Answer the question directly and concisely. Use the supplied research when it
is available. Do not claim facts that the research does not support.

Question:
{question.strip()}

Research:
{research or "No external research was supplied."}
"""
    return ChatOllama(model="hermes3", temperature=0).invoke(prompt).content.strip()
