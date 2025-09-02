"""
Thin compatibility wrapper: re-export LLMClient from verbatim_core.
"""

from verbatim_core.llm_client import LLMClient

__all__ = ["LLMClient"]
