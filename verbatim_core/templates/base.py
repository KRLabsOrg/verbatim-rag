"""Base template strategy interfaces (copy)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class TemplateStrategy(ABC):
    @abstractmethod
    def generate(self, question: str, spans: List[str], citation_count: int) -> str:
        pass

    @abstractmethod
    def fill(
        self,
        template: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
        pass

    def save_state(self) -> Dict[str, Any]:
        return {}

    def load_state(self, state: Dict[str, Any]) -> None:
        return None
