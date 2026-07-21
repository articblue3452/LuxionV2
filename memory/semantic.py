import json
from pathlib import Path
from typing import Any


class SemanticMemory:
    """Small JSON-backed store for stable user preferences."""

    def __init__(self) -> None:
        self.path = Path(__file__).with_name("semantic.json")

        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")

    def load(self) -> dict[str, Any]:
        data = json.loads(self.path.read_text(encoding="utf-8"))

        if not isinstance(data, dict):
            raise ValueError("Semantic memory must contain a JSON object.")

        return data

    def save(self, data: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, indent=4), encoding="utf-8")

    def get(self, key: str) -> Any | None:
        return self.load().get(key)

    def set(self, key: str, value: Any) -> None:
        data = self.load()
        data[key] = value
        self.save(data)

    def delete(self, key: str) -> None:
        data = self.load()

        if key in data:
            del data[key]

        self.save(data)

    def all(self) -> dict[str, Any]:
        return self.load()
