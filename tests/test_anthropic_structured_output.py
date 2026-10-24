"""Structured output on Claude models that reject a forced tool_choice.

Opus 5.5, Fable 5.1 and Sonnet 5.5 refuse ``tool_choice`` any/tool, so the
default function-calling method only suggests the tool and long prompts come
back as prose (parsed result None, free-text fallback every run). Those models
use Claude's native JSON-schema output; the rest keep forced tool calling.
"""

import pytest
from pydantic import BaseModel

from tradingagents.llm_clients.anthropic_client import AnthropicClient


class _Pick(BaseModel):
    rating: str


def _bound_kwargs(runnable):
    first = runnable.steps[0] if hasattr(runnable, "steps") else runnable
    return getattr(first, "kwargs", {})


@pytest.mark.unit
class TestAnthropicStructuredMethod:
    @pytest.mark.parametrize("model", ["claude-opus-5-5", "claude-fable-5-1", "claude-sonnet-5-5"])
    def test_models_without_forced_tool_choice_use_json_schema(self, model):
        bound = AnthropicClient(model=model, api_key="x").get_llm().with_structured_output(_Pick)
        kwargs = _bound_kwargs(bound)
        assert "tool_choice" not in kwargs and "tools" not in kwargs
        assert kwargs.get("output_config", {}).get("format", {}).get("type") == "json_schema"

    @pytest.mark.parametrize("model", ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"])
    def test_other_models_keep_forced_tool_calling(self, model):
        bound = AnthropicClient(model=model, api_key="x").get_llm().with_structured_output(_Pick)
        assert _bound_kwargs(bound).get("tool_choice", {}).get("name") == "_Pick"

    def test_explicit_method_is_respected(self):
        llm = AnthropicClient(model="claude-opus-5-5", api_key="x").get_llm()
        assert "tools" in _bound_kwargs(llm.with_structured_output(_Pick, method="function_calling"))
