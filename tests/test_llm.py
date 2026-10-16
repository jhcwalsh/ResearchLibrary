"""Tests for reading Claude responses on models that think by default."""
from types import SimpleNamespace

import pytest

from research_library import llm


def _msg(*blocks, stop_reason="end_turn"):
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, stop_details=None)


THINKING = SimpleNamespace(type="thinking", thinking="", signature="sig")


def test_model_ids_are_the_5_5_models():
    assert llm.HAIKU == "claude-haiku-5-5"
    assert llm.SONNET == "claude-sonnet-5-5"


def test_text_skips_a_leading_thinking_block():
    """5.5 models think by default, so content[0] is often an empty thinking block."""
    msg = _msg(THINKING, SimpleNamespace(type="text", text="the answer"))
    assert llm.response_text(msg) == "the answer"


def test_text_joins_multiple_text_blocks():
    msg = _msg(SimpleNamespace(type="text", text="a "), THINKING, SimpleNamespace(type="text", text="b"))
    assert llm.response_text(msg) == "a b"


def test_refusal_raises_a_clear_error():
    msg = _msg(SimpleNamespace(type="text", text=""), stop_reason="refusal")
    msg.stop_details = SimpleNamespace(category="general_harms", explanation="declined")
    with pytest.raises(llm.RefusalError, match="general_harms"):
        llm.response_text(msg)


def test_no_text_raises_and_names_the_stop_reason():
    """Thinking can use up max_tokens before any text is written."""
    with pytest.raises(RuntimeError, match="max_tokens"):
        llm.response_text(_msg(THINKING, stop_reason="max_tokens"))
