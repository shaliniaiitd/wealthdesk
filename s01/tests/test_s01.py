"""
s01/tests/test_s01.py
---------------------
Unit tests for Session 1: Basic Conversational Agent.

Run from the wealthdesk/ directory:
    pytest s01/tests/ -v

All tests run without a live Groq API key.
The LLM is mocked so tests are fast, deterministic, and safe to run in CI.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Point Python at the solution package before importing anything from it.
#
# SOLUTION_DIR resolves to s01/solution/ using this file's own path:
#   __file__           →  s01/tests/test_s01.py
#   .parent            →  s01/tests/
#   .parent.parent     →  s01/
#   / "solution"       →  s01/solution/
#
# The wealthdesk/ package sits inside s01/solution/, so inserting that
# directory at position 0 of sys.path makes "import wealthdesk" find the
# right copy — not some other session's version.
#
# The loop clears any previously loaded wealthdesk modules from Python's
# module cache (sys.modules). Without this, running the full test suite
# across multiple sessions (s01, s02, …) in one pytest invocation could
# leave an earlier session's wealthdesk loaded, and this file's imports
# would silently reuse that stale copy instead of s01/solution/.
# ---------------------------------------------------------------------------
SOLUTION_DIR = Path(__file__).parent.parent / "solution"
for _k in list(sys.modules):
    if _k == "wealthdesk" or _k.startswith("wealthdesk."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

# noqa: E402 silences the "module level import not at top of file" linter warning.
# These imports appear after sys.path.insert() above, which is intentional —
# the path must be set before Python can find the wealthdesk package.
from wealthdesk.config import MAX_TOKENS, MODEL_NAME, SYSTEM_PROMPT  # noqa: E402
from wealthdesk.state import WealthDeskState  # noqa: E402
import wealthdesk.nodes as _nodes  # noqa: E402
from wealthdesk.nodes import respond  # noqa: E402
from wealthdesk.agent import build_graph  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_response():
    # Replace the real `llm` object inside the nodes module with a fake that
    # returns a canned response. patch.object targets _nodes.llm specifically
    # so that any other code importing `llm` elsewhere is not affected.
    #
    # mock_result simulates what llm.invoke() normally returns: an object with
    # a .content attribute holding the model's reply text.
    #
    # `yield` hands the mock to the test so the test can inspect call args
    # (e.g. assert mock_llm.invoke.called). The patch is torn down automatically
    # when the test finishes, restoring the real llm.
    with patch.object(_nodes, "llm") as mock_llm:
        mock_result = MagicMock()
        mock_result.content = "The BNB home loan rate is 8.5% per annum. WealthDesk | Bharat National Bank"
        mock_llm.invoke.return_value = mock_result
        yield mock_llm


@pytest.fixture
def mock_llm_error():
    # Same as mock_llm_response but configures the fake llm to raise an
    # exception instead of returning a result. Tests that use this fixture
    # verify that respond() catches the exception and returns the safe
    # fallback string rather than crashing.
    with patch.object(_nodes, "llm") as mock_llm:
        mock_llm.invoke.side_effect = Exception("Groq API timeout")
        yield mock_llm


# ---------------------------------------------------------------------------
# State structure tests
#
# WealthDeskState is a TypedDict. Python automatically builds a dict called
# __annotations__ on every class that maps each declared field name (as a
# string) to its type:
#
#   class WealthDeskState(TypedDict):
#       customer_message: str      →   __annotations__ == {
#       response: str                      "customer_message": str,
#                                          "response": str
#                                      }
#
# The tests below check that exact name and exact type are both present.
# This matters because LangGraph reads state by exact string key at runtime —
# a typo like "customerMessage" or leaving `pass` in the TypedDict would
# silently break the graph rather than giving a clear error.
# ---------------------------------------------------------------------------

class TestWealthDeskState:
    def test_state_has_customer_message_field(self):
        assert "customer_message" in WealthDeskState.__annotations__

    def test_state_has_response_field(self):
        assert "response" in WealthDeskState.__annotations__

    def test_customer_message_is_str_type(self):
        assert WealthDeskState.__annotations__["customer_message"] is str

    def test_response_is_str_type(self):
        assert WealthDeskState.__annotations__["response"] is str

    def test_state_can_be_instantiated(self):
        state: WealthDeskState = {
            "customer_message": "What is the home loan rate?",
            "response": "",
        }
        assert state["customer_message"] == "What is the home loan rate?"
        assert state["response"] == ""


# ---------------------------------------------------------------------------
# System prompt tests
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_prompt_identifies_wealthdesk(self):
        assert "WealthDesk" in SYSTEM_PROMPT

    def test_prompt_identifies_bnb(self):
        assert "Bharat National Bank" in SYSTEM_PROMPT or "BNB" in SYSTEM_PROMPT

    def test_prompt_includes_out_of_scope_rule(self):
        lower = SYSTEM_PROMPT.lower()
        has_scope_rule = (
            "decline" in lower
            or "out-of-scope" in lower
            or "only discuss bnb" in lower
            or "only help with bnb" in lower
        )
        assert has_scope_rule

    def test_prompt_includes_home_loan_rate(self):
        assert "8.5" in SYSTEM_PROMPT

    def test_prompt_includes_fd_rate(self):
        assert "6.8" in SYSTEM_PROMPT or "7.1" in SYSTEM_PROMPT

    def test_model_name_is_not_empty(self):
        assert len(MODEL_NAME) > 0

    def test_max_tokens_is_reasonable(self):
        assert 100 < MAX_TOKENS <= 500


# ---------------------------------------------------------------------------
# respond() node tests
# ---------------------------------------------------------------------------

class TestRespondNode:
    def test_respond_returns_dict(self, mock_llm_response):
        state: WealthDeskState = {"customer_message": "What is the home loan rate?", "response": ""}
        result = respond(state)
        assert isinstance(result, dict)

    def test_respond_returns_response_key(self, mock_llm_response):
        state: WealthDeskState = {"customer_message": "Hello", "response": ""}
        result = respond(state)
        assert "response" in result

    def test_respond_content_is_string(self, mock_llm_response):
        state: WealthDeskState = {"customer_message": "What FDs do you offer?", "response": ""}
        result = respond(state)
        assert isinstance(result["response"], str)

    def test_respond_content_is_non_empty(self, mock_llm_response):
        state: WealthDeskState = {"customer_message": "Tell me about personal loans.", "response": ""}
        result = respond(state)
        assert len(result["response"]) > 0

    def test_respond_calls_llm_once(self, mock_llm_response):
        # MagicMock silently records every call made on it. assert_called_once()
        # checks that recorded history and fails if invoke() was called zero
        # times (forgot to call the LLM) or more than once (accidental loop).
        # This test does not check the return value — that is test_respond_returns_dict.
        # Each test checks exactly one thing.
        state: WealthDeskState = {"customer_message": "What is the FD rate?", "response": ""}
        respond(state)
        mock_llm_response.invoke.assert_called_once()

    def test_respond_passes_system_message(self, mock_llm_response):
        # call_args holds the arguments from the most recent invoke() call.
        # call_args[0] is the tuple of positional args; [0][0] is the first
        # positional arg — the messages list passed to llm.invoke(messages).
        # We then filter that list for SystemMessage instances and assert
        # exactly one exists (the SYSTEM_PROMPT from config.py).
        from langchain_core.messages import SystemMessage
        state: WealthDeskState = {"customer_message": "What is the FD rate?", "response": ""}
        respond(state)
        call_args = mock_llm_response.invoke.call_args
        messages = call_args[0][0]
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        assert len(system_messages) == 1

    def test_respond_passes_human_message_with_customer_text(self, mock_llm_response):
        # Same call_args[0][0] technique as above, but filtering for HumanMessage.
        # Also checks that .content matches the original customer question exactly —
        # catching bugs where the message is built but the text is lost or truncated.
        from langchain_core.messages import HumanMessage
        customer_question = "How much can I borrow for a home loan?"
        state: WealthDeskState = {"customer_message": customer_question, "response": ""}
        respond(state)
        call_args = mock_llm_response.invoke.call_args
        messages = call_args[0][0]
        human_messages = [m for m in messages if isinstance(m, HumanMessage)]
        assert len(human_messages) == 1
        assert human_messages[0].content == customer_question

    def test_respond_returns_safe_message_on_llm_error(self, mock_llm_error):
        # mock_llm_error sets side_effect on invoke(), so calling it raises
        # Exception("Groq API timeout") instead of returning a result.
        # This test checks that respond() catches that exception and still
        # returns a valid dict with a non-empty "response" string —
        # i.e. the try/except fallback path works correctly.
        state: WealthDeskState = {"customer_message": "What is the home loan rate?", "response": ""}
        result = respond(state)
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_respond_fallback_does_not_expose_error_details(self, mock_llm_error):
        # Checks that the exception message and class name do not leak into
        # the customer-facing response. Customers should never see a Python
        # traceback or internal error string — only the safe fallback text.
        state: WealthDeskState = {"customer_message": "test", "response": ""}
        result = respond(state)
        assert "Groq API timeout" not in result["response"]
        assert "Exception" not in result["response"]


# ---------------------------------------------------------------------------
# Graph structure tests
# ---------------------------------------------------------------------------

class TestGraph:
    def test_build_graph_returns_compiled_graph(self):
        graph = build_graph()
        assert graph is not None

    def test_graph_can_invoke_with_valid_state(self, mock_llm_response):
        graph = build_graph()
        state: WealthDeskState = {"customer_message": "What is the home loan rate?", "response": ""}
        result = graph.invoke(state)
        assert "response" in result

    def test_graph_invocation_returns_string_response(self, mock_llm_response):
        graph = build_graph()
        state: WealthDeskState = {"customer_message": "Tell me about gold loans.", "response": ""}
        result = graph.invoke(state)
        assert isinstance(result["response"], str)

    def test_graph_preserves_customer_message(self, mock_llm_response):
        # respond() only returns {"response": ...} — it does not return
        # customer_message. LangGraph merges that partial dict back into the
        # full state snapshot, so customer_message must still be present in
        # the result. This test verifies that LangGraph's merge behaviour
        # works and that respond() does not accidentally drop or overwrite
        # fields it did not touch.
        graph = build_graph()
        question = "What is the personal loan interest rate?"
        state: WealthDeskState = {"customer_message": question, "response": ""}
        result = graph.invoke(state)
        assert result["customer_message"] == question
