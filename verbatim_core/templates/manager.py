"""
Template manager (RAG-agnostic copy) managing strategy selection and processing.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Any, List, Optional

from .base import TemplateStrategy
from .static import StaticTemplate
from .contextual import ContextualTemplate
from .random import RandomTemplate
from verbatim_rag.llm_client import LLMClient


class TemplateManager:
    def __init__(
        self, llm_client: Optional[LLMClient] = None, default_mode: str = "static"
    ):
        self.llm_client = llm_client
        self.current_mode = default_mode

        self.strategies: Dict[str, TemplateStrategy | None] = {
            "static": StaticTemplate(),
            "contextual": ContextualTemplate(llm_client) if llm_client else None,
            "random": RandomTemplate(llm_client=llm_client),
        }

        if self.current_mode not in self.strategies:
            self.current_mode = "static"
        if self.strategies[self.current_mode] is None:
            print(
                f"Warning: {self.current_mode} mode requires LLM client, falling back to static"
            )
            self.current_mode = "static"

    def set_mode(self, mode: str) -> bool:
        if mode not in self.strategies:
            print(f"Unknown template mode: {mode}")
            return False
        if self.strategies[mode] is None:
            print(f"Mode {mode} is not available (requires LLM client)")
            return False
        self.current_mode = mode
        return True

    def get_current_mode(self) -> str:
        return self.current_mode

    def get_available_modes(self) -> List[str]:
        return [
            mode for mode, strategy in self.strategies.items() if strategy is not None
        ]

    def process(
        self,
        question: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
        all_spans = [span["text"] for span in display_spans + citation_spans]
        citation_count = len(citation_spans)
        strategy = self.strategies[self.current_mode]
        assert strategy is not None
        template = strategy.generate(question, all_spans, citation_count)
        return strategy.fill(template, display_spans, citation_spans)

    async def process_async(
        self,
        question: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
        all_spans = [span["text"] for span in display_spans + citation_spans]
        citation_count = len(citation_spans)
        strategy = self.strategies[self.current_mode]
        assert strategy is not None
        if hasattr(strategy, "generate_async") and self.current_mode == "contextual":
            template = await strategy.generate_async(
                question, all_spans, citation_count
            )  # type: ignore[attr-defined]
        else:
            template = strategy.generate(question, all_spans, citation_count)
        return strategy.fill(template, display_spans, citation_spans)

    def get_template(
        self,
        question: str = "",
        spans: List[str] | None = None,
        citation_count: int = 0,
    ) -> str:
        spans = spans or []
        strategy = self.strategies[self.current_mode]
        assert strategy is not None
        return strategy.generate(question, spans, citation_count)

    def fill_template(
        self,
        template: str,
        display_spans: List[Dict[str, Any]],
        citation_spans: List[Dict[str, Any]],
    ) -> str:
        strategy = self.strategies[self.current_mode]
        assert strategy is not None
        return strategy.fill(template, display_spans, citation_spans)

    def save(self, filepath: str) -> None:
        data = {"current_mode": self.current_mode, "strategies": {}}
        for mode, strategy in self.strategies.items():
            if strategy is not None:
                data["strategies"][mode] = strategy.save_state()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> bool:
        if not os.path.exists(filepath):
            print(f"Template config file not found: {filepath}")
            return False
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            if "current_mode" in data:
                mode = data["current_mode"]
                if self.strategies.get(mode) is not None:
                    self.current_mode = mode
            strategies_data = data.get("strategies", {})
            for mode, state in strategies_data.items():
                if mode in self.strategies and self.strategies[mode] is not None:
                    try:
                        self.strategies[mode].load_state(state)  # type: ignore[union-attr]
                    except Exception as e:
                        print(f"Warning: Failed to load state for {mode} strategy: {e}")
            return True
        except Exception as e:
            print(f"Failed to load template config: {e}")
            return False

    def info(self) -> Dict[str, Any]:
        return {
            "current_mode": self.current_mode,
            "available_modes": self.get_available_modes(),
        }
