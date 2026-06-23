"""
s02/tests/test_s02.py
---------------------
Unit tests for Session 2: Multi-Turn Memory.

Run from the wealthdesk/ directory:
    pytest s02/tests/ -v

All tests run without a live Groq API key or a real database file.
The LLM is mocked. Memory tests use an in-memory SqliteSaver so the
filesystem is never touched.

Design rationale:
  - State structure tests: catch regressions if someone removes 'history'
    when adding new fields in later sessions.
  - respond() tests: verify the node builds messages from history correctly
    and updates history after each turn.
  - Memory persistence tests: verify that the same thread_id loads previous
    history, while a different thread_id starts fresh. This is the core
    behavioural contract of US-02.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

sys.modules.pop("main", None)

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
sys.path.insert(0, str(SOLUTION_DIR))

from main import (  # noqa: E402
    WealthDeskState,
    SYSTEM_PROMPT,
    build_graph,
    respond,
)
import main as _s02_module  # noqa: E402  (keep a direct reference so patch.object works even when sys.modules["main"] is overwritten by another session's import)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_checkpointer():
    """In-memory checkpointer for test isolation.

    Each test that uses this fixture gets an isolated, empty checkpointer.
    No file is written to disk. MemorySaver stores state in a plain dict,
    which is sufficient to verify all memory persistence behaviour:
    same thread_id shares history; different thread_ids are independent.

    Production code uses SqliteSaver so state survives script restarts.
    MemorySaver is correct for tests because they do not need persistence
    beyond a single test run.
    """
    return MemorySaver()


@pytest.fixture
def mock_llm_response():
    """Patch ChatGroq so tests never make a real API call.

    Returns a fixed response string. Tests that need specific content can
    override mock_llm.invoke.return_value.content inside the test body.
    """
    with patch.object(_s02_module, "llm") as mock_llm:
        mock_result = MagicMock()
        mock_result.content = "The BNB home loan rate is 8.5% per annum. WealthDesk | Bharat National Bank"
        mock_llm.invoke.return_value = mock_result
        yield mock_llm


@pytest.fixture
def mock_llm_error():
    """Patch ChatGroq to raise an exception, testing the error-handling path."""
    with patch.object(_s02_module, "llm") as mock_llm:
        mock_llm.invoke.side_effect = Exception("Groq API timeout")
        yield mock_llm


# ---------------------------------------------------------------------------
# State structure tests
# ---------------------------------------------------------------------------

class TestWealthDeskState:
    """Verify the TypedDict definition has the shape required for Session 2."""

    def test_state_has_customer_message_field(self):
        assert "customer_message" in WealthDeskState.__annotations__

    def test_state_has_response_field(self):
        assert "response" in WealthDeskState.__annotations__

    def test_state_has_history_field(self):
        """Session 2 adds 'history'. This test catches regressions in later sessions."""
        assert "history" in WealthDeskState.__annotations__, (
            "WealthDeskState must have a 'history' field (added in Session 2). "
            "Check that you did not accidentally remove it when editing the TypedDict."
        )

    def test_customer_message_is_str(self):
        assert WealthDeskState.__annotations__["customer_message"] is str

    def test_response_is_str(self):
        assert WealthDeskState.__annotations__["response"] is str

    def test_history_is_list_type(self):
        """history must be typed as list[dict], not str or any other type."""
        annotation = WealthDeskState.__annotations__["history"]
        # list[dict] is a GenericAlias -- check the origin is list
        import types
        origin = getattr(annotation, "__origin__", annotation)
        assert origin is list, (
            f"'history' should be typed as list[dict] but found: {annotation}"
        )

    def test_state_can_be_instantiated_with_history(self):
        state: WealthDeskState = {
            "customer_message": "What is the home loan rate?",
            "response":         "",
            "history":          [],
        }
        assert state["history"] == []

    def test_state_history_accepts_turn_dicts(self):
        turns = [
            {"role": "user",      "content": "What is the home loan rate?"},
            {"role": "assistant", "content": "8.5% p.a."},
        ]
        state: WealthDeskState = {
            "customer_message": "And the FD rate?",
            "response":         "",
            "history":          turns,
        }
        assert len(state["history"]) == 2
        assert state["history"][0]["role"] == "user"


# ---------------------------------------------------------------------------
# respond() node tests
# ---------------------------------------------------------------------------

class TestRespondNode:
    """Test the respond() node in isolation."""

    def test_respond_returns_dict(self, mock_llm_response):
        state: WealthDeskState = {
            "customer_message": "What is the home loan rate?",
            "response":         "",
            "history":          [],
        }
        result = respond(state)
        assert isinstance(result, dict)

    def test_respond_returns_response_key(self, mock_llm_response):
        state: WealthDeskState = {
            "customer_message": "Hello",
            "response":         "",
            "history":          [],
        }
        result = respond(state)
        assert "response" in result

    def test_respond_returns_history_key(self, mock_llm_response):
        """Session 2: respond() must return an updated 'history' key."""
        state: WealthDeskState = {
            "customer_message": "What is the FD rate?",
            "response":         "",
            "history":          [],
        }
        result = respond(state)
        assert "history" in result, (
            "respond() must return a 'history' key with the updated conversation. "
            "Check that your return statement includes 'history': new_history."
        )

    def test_respond_appends_user_turn_to_history(self, mock_llm_response):
        """The customer's message must appear in history after the node runs."""
        question = "What is the home loan interest rate?"
        state: WealthDeskState = {
            "customer_message": question,
            "response":         "",
            "history":          [],
        }
        result = respond(state)
        user_turns = [t for t in result["history"] if t["role"] == "user"]
        assert any(t["content"] == question for t in user_turns), (
            "After respond() runs, history must contain a user turn "
            "with the original customer message."
        )

    def test_respond_appends_assistant_turn_to_history(self, mock_llm_response):
        """The agent's response must appear in history after the node runs."""
        state: WealthDeskState = {
            "customer_message": "What FDs do you offer?",
            "response":         "",
            "history":          [],
        }
        result = respond(state)
        assistant_turns = [t for t in result["history"] if t["role"] == "assistant"]
        assert len(assistant_turns) >= 1, (
            "After respond() runs, history must contain at least one assistant turn."
        )

    def test_respond_history_grows_by_two_per_turn(self, mock_llm_response):
        """Each call to respond() adds exactly two items: one user, one assistant."""
        state: WealthDeskState = {
            "customer_message": "Tell me about personal loans.",
            "response":         "",
            "history":          [],
        }
        result = respond(state)
        assert len(result["history"]) == 2, (
            "A respond() call on an empty history should produce a history of length 2 "
            "(one user turn + one assistant turn)."
        )

    def test_respond_preserves_previous_history(self, mock_llm_response):
        """respond() must append to history, not replace it."""
        previous_turns = [
            {"role": "user",      "content": "What is the home loan rate?"},
            {"role": "assistant", "content": "8.5% p.a."},
        ]
        state: WealthDeskState = {
            "customer_message": "And the FD rate for 2 years?",
            "response":         "",
            "history":          previous_turns,
        }
        result = respond(state)
        assert len(result["history"]) == 4, (
            "respond() must append two new turns to the existing history. "
            "If previous history had 2 items, the result should have 4."
        )
        # First two entries must be the original turns, unchanged
        assert result["history"][0] == previous_turns[0]
        assert result["history"][1] == previous_turns[1]

    def test_respond_includes_history_in_llm_call(self, mock_llm_response):
        """The LLM must receive previous turns as HumanMessage/AIMessage pairs."""
        from langchain_core.messages import AIMessage, HumanMessage

        previous_turns = [
            {"role": "user",      "content": "I earn Rs. 80,000 per month."},
            {"role": "assistant", "content": "Noted. How can I help you?"},
        ]
        state: WealthDeskState = {
            "customer_message": "How much home loan can I get?",
            "response":         "",
            "history":          previous_turns,
        }
        respond(state)

        call_args = mock_llm_response.invoke.call_args
        messages  = call_args[0][0]

        # The previous user turn must appear in the messages
        human_contents = [m.content for m in messages if isinstance(m, HumanMessage)]
        assert "I earn Rs. 80,000 per month." in human_contents, (
            "respond() must include previous user turns as HumanMessages in the LLM call. "
            "The LLM cannot answer 'how much can I get?' without knowing the income mentioned earlier."
        )

        # The previous assistant turn must appear in the messages
        ai_contents = [m.content for m in messages if isinstance(m, AIMessage)]
        assert "Noted. How can I help you?" in ai_contents, (
            "respond() must include previous assistant turns as AIMessages in the LLM call."
        )

    def test_respond_passes_system_message(self, mock_llm_response):
        from langchain_core.messages import SystemMessage

        state: WealthDeskState = {
            "customer_message": "What is the FD rate?",
            "response":         "",
            "history":          [],
        }
        respond(state)

        call_args = mock_llm_response.invoke.call_args
        messages  = call_args[0][0]
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        assert len(system_msgs) == 1

    def test_respond_returns_safe_message_on_llm_error(self, mock_llm_error):
        """When the LLM raises, respond() must not propagate the exception."""
        state: WealthDeskState = {
            "customer_message": "What is the home loan rate?",
            "response":         "",
            "history":          [],
        }
        result = respond(state)
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_respond_fallback_does_not_expose_error_details(self, mock_llm_error):
        state: WealthDeskState = {
            "customer_message": "test",
            "response":         "",
            "history":          [],
        }
        result = respond(state)
        assert "Groq API timeout" not in result["response"]
        assert "Exception" not in result["response"]

    def test_respond_error_still_updates_history(self, mock_llm_error):
        """Even on LLM error, history must include both sides of the exchange."""
        state: WealthDeskState = {
            "customer_message": "What is the home loan rate?",
            "response":         "",
            "history":          [],
        }
        result = respond(state)
        # History should still grow -- the fallback response is the assistant turn
        assert "history" in result
        assert len(result["history"]) == 2


# ---------------------------------------------------------------------------
# Memory persistence tests
# ---------------------------------------------------------------------------

class TestMemoryPersistence:
    """Verify that the checkpointer correctly stores and retrieves history.

    These tests are the core of US-02. They confirm the behavioural contract:
    same thread_id = shared memory across invocations.
    different thread_id = separate, isolated memory.
    """

    def test_same_thread_id_shares_history(self, mock_llm_response, memory_checkpointer):
        """Two graph.invoke() calls with the same thread_id share history."""
        graph     = build_graph(checkpointer=memory_checkpointer)
        thread_id = "test-thread-001"
        config    = {"configurable": {"thread_id": thread_id}}

        # Turn 1: ask about income
        mock_llm_response.invoke.return_value.content = (
            "Understood. With an income of Rs. 80,000 per month, "
            "you may be eligible for a significant home loan."
        )
        graph.invoke(
            {"customer_message": "I earn Rs. 80,000 per month.", "response": ""},
            config=config,
        )

        # Turn 2: ask a follow-up that only makes sense in context
        mock_llm_response.invoke.return_value.content = (
            "Based on your income of Rs. 80,000, you can get up to Rs. 48 lakhs."
        )
        result = graph.invoke(
            {"customer_message": "How much home loan can I get?", "response": ""},
            config=config,
        )

        # The LLM must have received the income mention from Turn 1
        call_args  = mock_llm_response.invoke.call_args
        messages   = call_args[0][0]
        all_content = " ".join(m.content for m in messages)
        assert "80,000" in all_content or "Rs. 80" in all_content, (
            "On the second invocation with the same thread_id, the LLM should "
            "receive Turn 1's income detail in the message history. "
            "Confirm that build_graph() compiles with the checkpointer."
        )

    def test_different_thread_ids_have_separate_histories(
        self, mock_llm_response, memory_checkpointer
    ):
        """Two thread_ids must not share state."""
        graph  = build_graph(checkpointer=memory_checkpointer)
        config_a = {"configurable": {"thread_id": "thread-A"}}
        config_b = {"configurable": {"thread_id": "thread-B"}}

        # Thread A: turn 1
        mock_llm_response.invoke.return_value.content = "Home loan info for thread A."
        graph.invoke(
            {"customer_message": "What is the home loan rate?", "response": ""},
            config=config_a,
        )

        # Thread B: turn 1 -- independent conversation
        mock_llm_response.invoke.return_value.content = "FD info for thread B."
        graph.invoke(
            {"customer_message": "Tell me about FDs.", "response": ""},
            config=config_b,
        )

        # Thread B: turn 2 -- LLM should NOT see Thread A's question
        mock_llm_response.invoke.return_value.content = "More FD info."
        graph.invoke(
            {"customer_message": "What are the FD rates?", "response": ""},
            config=config_b,
        )

        call_args  = mock_llm_response.invoke.call_args
        messages   = call_args[0][0]
        all_content = " ".join(m.content for m in messages)
        assert "home loan rate" not in all_content.lower(), (
            "Thread B's second turn should not contain Thread A's question. "
            "Different thread_ids must have completely separate histories."
        )

    def test_history_survives_fresh_graph_invocation(self, mock_llm_response, memory_checkpointer):
        """A second graph.invoke() with the same thread_id loads the saved state."""
        graph     = build_graph(checkpointer=memory_checkpointer)
        thread_id = "test-thread-002"
        config    = {"configurable": {"thread_id": thread_id}}

        mock_llm_response.invoke.return_value.content = "Personal loan response."
        result_turn1 = graph.invoke(
            {"customer_message": "Tell me about personal loans.", "response": ""},
            config=config,
        )
        history_after_turn1 = result_turn1.get("history", [])

        mock_llm_response.invoke.return_value.content = "Second response."
        result_turn2 = graph.invoke(
            {"customer_message": "What is the processing fee?", "response": ""},
            config=config,
        )
        history_after_turn2 = result_turn2.get("history", [])

        assert len(history_after_turn2) > len(history_after_turn1), (
            "History should grow with each turn. After two turns it should be longer "
            "than after one turn, which confirms the checkpointer is accumulating state."
        )

    def test_invoke_without_config_raises_or_discards_history(self, mock_llm_response, memory_checkpointer):
        """Invoking without config is the 'forgot the thread_id' mistake.

        This test documents the behaviour but does not assert a specific outcome --
        some LangGraph versions raise, others silently start a new thread.
        The important lesson is: always pass config.
        """
        graph = build_graph(checkpointer=memory_checkpointer)
        try:
            result = graph.invoke(
                {"customer_message": "Hello", "response": ""},
                # No config -- no thread_id
            )
            # If it does not raise, that is acceptable behaviour to document
            assert "response" in result
        except Exception:
            # Also acceptable -- some versions require config when checkpointer is present
            pass


# ---------------------------------------------------------------------------
# Graph structure tests
# ---------------------------------------------------------------------------

class TestGraph:
    """Verify the compiled graph structure for Session 2."""

    def test_build_graph_accepts_checkpointer_argument(self, memory_checkpointer):
        """build_graph() must accept an optional checkpointer parameter."""
        graph = build_graph(checkpointer=memory_checkpointer)
        assert graph is not None

    def test_build_graph_with_no_args_does_not_crash(self):
        """build_graph() with no arguments should use the file-based checkpointer.

        This test does NOT run the graph (that would create a real .db file)
        -- it only verifies the graph compiles without error.
        We patch sqlite3.connect so no filesystem write occurs.
        """
        import sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        with patch("main.sqlite3") as mock_sqlite3, \
             patch("main.SqliteSaver") as mock_saver_class:
            mock_conn = MagicMock(spec=sqlite3.Connection)
            mock_sqlite3.connect.return_value = mock_conn
            mock_saver_class.return_value = MemorySaver()
            graph = build_graph()
            assert graph is not None

    def test_graph_invocation_returns_response(self, mock_llm_response, memory_checkpointer):
        graph  = build_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "test-basic"}}
        state: WealthDeskState = {
            "customer_message": "What is the home loan rate?",
            "response":         "",
            "history":          [],
        }
        result = graph.invoke(state, config=config)
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_graph_invocation_returns_history(self, mock_llm_response, memory_checkpointer):
        """After graph.invoke(), state must contain an updated history."""
        graph  = build_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "test-history-return"}}
        state: WealthDeskState = {
            "customer_message": "Tell me about gold loans.",
            "response":         "",
            "history":          [],
        }
        result = graph.invoke(state, config=config)
        assert "history" in result
        assert isinstance(result["history"], list)
        assert len(result["history"]) >= 2

    def test_graph_preserves_customer_message(self, mock_llm_response, memory_checkpointer):
        graph    = build_graph(checkpointer=memory_checkpointer)
        config   = {"configurable": {"thread_id": "test-preservation"}}
        question = "What is the personal loan interest rate?"
        state: WealthDeskState = {
            "customer_message": question,
            "response":         "",
            "history":          [],
        }
        result = graph.invoke(state, config=config)
        assert result["customer_message"] == question
