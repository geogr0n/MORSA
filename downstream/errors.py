from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MissingItem:
    path: str
    reason: str
    context: str = ""


class MissingInputError(RuntimeError):
    def __init__(self, message: str, missing: list[MissingItem] | None = None) -> None:
        super().__init__(message)
        self.missing = missing or []

    def __str__(self) -> str:
        message = super().__str__()
        if not self.missing:
            return message
        details = "\n".join(
            f"- {item.path}: {item.reason}"
            + (f" ({item.context})" if item.context else "")
            for item in self.missing
        )
        return f"{message}\n{details}"


class ResourceError(MissingInputError):
    pass
