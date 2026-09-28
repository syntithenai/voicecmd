"""Intent / reply types shared by the router, handlers and LLM tool calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Intent:
    domain: str  # music | timer | weather | clock | system
    action: str
    params: dict[str, Any] = field(default_factory=dict)
    source: str = "regex"  # regex | llm


@dataclass
class Reply:
    text: str
    ok: bool = True
    intent: Intent | None = None
    expects_reply: bool = False
    silent: bool = False
    data: dict[str, Any] = field(default_factory=dict)
