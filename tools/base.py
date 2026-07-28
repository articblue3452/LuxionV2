"""Small, framework-independent contract shared by every Luxion tool."""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    function: Callable[..., Any]
    parameters: list[str]
    optional_parameters: list[str] = field(default_factory=list)

    def validate_args(self, args: Any) -> str | None:
        """Return a precondition error, never execute with incomplete input."""
        if not isinstance(args, dict):
            return "Tool arguments must be a JSON object."
        missing = [key for key in self.parameters if key not in args or args[key] in (None, "")]
        if missing:
            return f"Missing required inputs: {', '.join(missing)}."
        allowed = set(self.parameters) | set(self.optional_parameters)
        unknown = set(args) - allowed
        if unknown:
            return f"Unknown inputs: {', '.join(sorted(unknown))}."
        return None
