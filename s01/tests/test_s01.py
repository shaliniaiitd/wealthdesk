"""
s01/tests/test_s01.py
---------------------
Unit tests for Session 1: Basic Conversational Agent.

Run from the wealthdesk/ directory:
    pytest s01/tests/ -v

All tests run without a live Groq API key.
The LLM is mocked so tests are fast, deterministic, and safe to run in CI.

Design rationale for each test:
  - State and graph structure tests: catch regressions when fields are added/removed
    in later sessions. If a participant accidentally deletes a field, the test
    fails before runtime.
  - Node tests with mocked LLM: verify the node's logic (message construction,
    error handling, return shape) independently of what the LLM actually says.
  - System prompt tests: verify the persona and rules are present without
    testing the exact wording (which changes as the course evolves).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# When pytest runs multiple sessions in one command, 'main' from an earlier
# session stays in sys.modules. Clear it before inserting this session's path
# so this test file always imports from its own solution/ directory.
sys.modules.pop("main", None)

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
sys.path.insert(0, str(SOLUTION_DIR))

from main import (  # noqa: E402  (import after sys.path modification)
    WealthDeskState,
    SYSTEM_PROMPT,
    MODEL_NAME,
    MAX_TOKENS,
    respond,
    build_graph,
)
import main as _s01_module  # noqa: E402  (keep a direct reference so patch.object works even when sys.modules["main"] is overwritten by a later session's import)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_response():
    """Patch ChatGroq so tests never make a real API call.

    The mock returns a MagicMock with a .content attribute (matching the
    actual ChatGroq response shape). Every test that calls respond() should
    use this fixture or the 'mock_groq_error' fixture.
    """
    with patch.object(_s01_module, "llm") as mock_llm:
        mock_result = MagicMock()
        mock_result.content = "The BNB home loan rate is 8.5% per annum. WealthDesk | Bharat National Bank"
        mock_llm.invoke.return_value = mock_result
        yield mock_llm


@pytest.fixture
def mock_llm_error():
    """Patch ChatGroq to raise an exception, testing the error-handling path."""
    with patch.object(_s01_module, "llm") as mock_llm:
        mock_llm.invoke.side_effect = Exception("Groq API timeout")
        yield mock_llm


# ---------------------------------------------------------------------------
# State structure tests
# ---------------------------------------------------------------------------

class TestWealthDeskState:
    """Verify the TypedDict definition has the required shape for Session 1."""

    def test_state_has_customer_message_field(self):
        """WealthDeskState must have a customer_message field."""
        assert "customer_message" in WealthDeskState.__annotations__

    def test_state_has_response_field(self):
        """WealthDeskState must have a response field."""
        assert "response" in WealthDeskState.__annotations__

    def test_customer_message_is_str_type(self):
        assert WealthDeskState.__annotations__["customer_message"] is str

    def test_response_is_str_type(self):
        assert WealthDeskState.__annotations__["response"] is str

    def test_state_can_be_instantiated(self):
        """A WealthDeskState dict should be creatable with the two required keys."""
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
    """Verify the persona and rules are encoded in the system prompt."""

    def test_prompt_identifies_wealthdesk(self):
        """The agent must identify itself as WealthDesk."""
        assert "WealthDesk" in SYSTEM_PROMPT

    def test_prompt_identifies_bnb(self):
        """The agent must identify the bank as BNB or Bharat National Bank."""
        assert "Bharat National Bank" in SYSTEM_PROMPT or "BNB" in SYSTEM_PROMPT

    def test_prompt_includes_out_of_scope_rule(self):
        """The prompt must explicitly instruct the model to decline out-of-scope requests."""
        lower = SYSTEM_PROMPT.lower()
        # Check for a specific decline instruction, not just the word "only"
        # (which appears in many unrelated contexts)
        has_scope_rule = (
            "decline" in lower
            or "out-of-scope" in lower
            or "only discuss bnb" in lower
            or "only help with bnb" in lower
        )
        assert has_scope_rule, (
            "System prompt must contain an explicit rule to decline out-of-scope requests. "
            "Expected one of: 'Decline', 'out-of-scope', 'only discuss BNB', 'only help with BNB'."
        )

    def test_prompt_includes_home_loan_rate(self):
        """Home loan rate (8.5%) must appear in the system prompt for Session 1."""
        assert "8.5" in SYSTEM_PROMPT

    def test_prompt_includes_fd_rate(self):
        """At least one FD rate must appear in the system prompt for Session 1."""
        assert "6.8" in SYSTEM_PROMPT or "7.1" in SYSTEM_PROMPT

    def test_model_name_is_not_empty(self):
        assert len(MODEL_NAME) > 0

    def test_max_tokens_is_reasonable(self):
        """150 words is ~200 tokens. 300 is a reasonable ceiling."""
        assert 100 < MAX_TOKENS <= 500


# ---------------------------------------------------------------------------
# respond() node tests
# ---------------------------------------------------------------------------

class TestRespondNode:
    """Test the respond() node in isolation."""

    def test_respond_returns_dict(self, mock_llm_response):
        state: WealthDeskState = {
            "customer_message": "What is the home loan rate?",
            "response": "",
        }
        result = respond(state)
        assert isinstance(result, dict)

    def test_respond_returns_response_key(self, mock_llm_response):
        """The node must always return a dict containing the 'response' key."""
        state: WealthDeskState = {
            "customer_message": "Hello",
            "response": "",
        }
        result = respond(state)
        assert "response" in result

    def test_respond_content_is_string(self, mock_llm_response):
        state: WealthDeskState = {
            "customer_message": "What FDs do you offer?",
            "response": "",
        }
        result = respond(state)
        assert isinstance(result["response"], str)

    def test_respond_content_is_non_empty(self, mock_llm_response):
        """A successful LLM call must produce a non-empty response."""
        state: WealthDeskState = {
            "customer_message": "Tell me about personal loans.",
            "response": "",
        }
        result = respond(state)
        assert len(result["response"]) > 0

    def test_respond_calls_llm_once(self, mock_llm_response):
        """Each call to respond() should invoke the LLM exactly once."""
        state: WealthDeskState = {
            "customer_message": "What is the FD rate?",
            "response": "",
        }
        respond(state)
        mock_llm_response.invoke.assert_called_once()

    def test_respond_passes_system_message(self, mock_llm_response):
        """respond() must include a SystemMessage in the messages it sends to the LLM."""
        from langchain_core.messages import SystemMessage

        state: WealthDeskState = {
            "customer_message": "What is the FD rate?",
            "response": "",
        }
        respond(state)

        call_args = mock_llm_response.invoke.call_args
        messages = call_args[0][0]  # first positional arg to invoke()
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        assert len(system_messages) == 1

    def test_respond_passes_human_message_with_customer_text(self, mock_llm_response):
        """The customer's question must appear in a HumanMessage."""
        from langchain_core.messages import HumanMessage

        customer_question = "How much can I borrow for a home loan?"
        state: WealthDeskState = {
            "customer_message": customer_question,
            "response": "",
        }
        respond(state)

        call_args = mock_llm_response.invoke.call_args
        messages = call_args[0][0]
        human_messages = [m for m in messages if isinstance(m, HumanMessage)]
        assert len(human_messages) == 1
        assert human_messages[0].content == customer_question

    def test_respond_returns_safe_message_on_llm_error(self, mock_llm_error):
        """When the LLM raises an exception, respond() must not propagate it."""
        state: WealthDeskState = {
            "customer_message": "What is the home loan rate?",
            "response": "",
        }
        result = respond(state)
        # The node must return a dict with a 'response' key, not raise
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_respond_fallback_does_not_expose_error_details(self, mock_llm_error):
        """The fallback message must not contain internal error details."""
        state: WealthDeskState = {
            "customer_message": "test",
            "response": "",
        }
        result = respond(state)
        # These strings from the mock exception must not reach the customer
        assert "Groq API timeout" not in result["response"]
        assert "Exception" not in result["response"]


# ---------------------------------------------------------------------------
# Graph structure tests
# ---------------------------------------------------------------------------

class TestGraph:
    """Verify the LangGraph graph compiles and has the expected structure."""

    def test_build_graph_returns_compiled_graph(self):
        """build_graph() must return a compiled graph object (not None, not a builder)."""
        graph = build_graph()
        assert graph is not None

    def test_graph_can_invoke_with_valid_state(self, mock_llm_response):
        """The compiled graph must be invocable with a WealthDeskState dict."""
        graph = build_graph()
        state: WealthDeskState = {
            "customer_message": "What is the home loan rate?",
            "response": "",
        }
        result = graph.invoke(state)
        assert "response" in result

    def test_graph_invocation_returns_string_response(self, mock_llm_response):
        graph = build_graph()
        state: WealthDeskState = {
            "customer_message": "Tell me about gold loans.",
            "response": "",
        }
        result = graph.invoke(state)
        assert isinstance(result["response"], str)

    def test_graph_preserves_customer_message(self, mock_llm_response):
        """graph.invoke() must return the original customer_message unchanged."""
        graph = build_graph()
        question = "What is the personal loan interest rate?"
        state: WealthDeskState = {
            "customer_message": question,
            "response": "",
        }
        result = graph.invoke(state)
        assert result["customer_message"] == question
