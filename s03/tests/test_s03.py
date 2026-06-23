"""
s03/tests/test_s03.py
---------------------
Unit tests for Session 3: Query Routing.

Run from the wealthdesk/ directory:
    pytest s03/tests/ -v

All tests run without a live Groq API key.
The LLM is mocked. classify() and respond() are tested in isolation.
End-to-end routing tests use MemorySaver to verify the full graph path.

Design rationale:
  - State tests: catch regressions if query_type field is removed later.
  - classify() tests: verify correct parsing of LLM output and safe defaults
    for unexpected or erroneous classifier responses.
  - route_query() tests: every branch, deterministic, no LLM needed.
  - escalate() / decline() tests: verify no LLM call and correct history update.
  - Graph routing tests: invoke the compiled graph and verify the correct node
    ran by inspecting query_type in the result state.
  - Memory tests: confirm routing still accumulates history correctly.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver

sys.modules.pop("main", None)

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
sys.path.insert(0, str(SOLUTION_DIR))

from main import (  # noqa: E402
    DECLINE_RESPONSE,
    ESCALATE_RESPONSE,
    WealthDeskState,
    build_graph,
    classify,
    decline,
    escalate,
    respond,
    route_query,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_checkpointer():
    return MemorySaver()


@pytest.fixture
def mock_llm_simple():
    """Patch both llm and classifier_llm to simulate a SIMPLE classification
    and a normal respond() answer."""
    with patch("main.llm") as mock_main, patch("main.classifier_llm") as mock_clf:
        mock_clf_result = MagicMock()
        mock_clf_result.content = "SIMPLE"
        mock_clf.invoke.return_value = mock_clf_result

        mock_main_result = MagicMock()
        mock_main_result.content = "The BNB home loan rate is 8.5% p.a. WealthDesk | Bharat National Bank"
        mock_main.invoke.return_value = mock_main_result

        yield mock_main, mock_clf


@pytest.fixture
def mock_llm_complex():
    """Classifier returns COMPLEX; main llm not expected to be called."""
    with patch("main.classifier_llm") as mock_clf:
        mock_result = MagicMock()
        mock_result.content = "COMPLEX"
        mock_clf.invoke.return_value = mock_result
        yield mock_clf


@pytest.fixture
def mock_llm_out_of_scope():
    """Classifier returns OUT_OF_SCOPE; main llm not expected to be called."""
    with patch("main.classifier_llm") as mock_clf:
        mock_result = MagicMock()
        mock_result.content = "OUT_OF_SCOPE"
        mock_clf.invoke.return_value = mock_result
        yield mock_clf


@pytest.fixture
def mock_classifier_error():
    """Classifier raises an exception."""
    with patch("main.classifier_llm") as mock_clf:
        mock_clf.invoke.side_effect = Exception("Groq API timeout")
        yield mock_clf


# ---------------------------------------------------------------------------
# State structure tests
# ---------------------------------------------------------------------------

class TestWealthDeskState:
    """Session 3 must add query_type to the state TypedDict."""

    def test_state_has_customer_message(self):
        assert "customer_message" in WealthDeskState.__annotations__

    def test_state_has_response(self):
        assert "response" in WealthDeskState.__annotations__

    def test_state_has_history(self):
        assert "history" in WealthDeskState.__annotations__

    def test_state_has_query_type(self):
        """query_type must be present -- it is the routing key for this session."""
        assert "query_type" in WealthDeskState.__annotations__, (
            "WealthDeskState must have a 'query_type' field (added in Session 3). "
            "Add it after 'history' with type hint str."
        )

    def test_query_type_is_str(self):
        assert WealthDeskState.__annotations__["query_type"] is str

    def test_state_instantiable_with_all_fields(self):
        state: WealthDeskState = {
            "customer_message": "What is the FD rate?",
            "response":         "",
            "history":          [],
            "query_type":       "SIMPLE",
        }
        assert state["query_type"] == "SIMPLE"


# ---------------------------------------------------------------------------
# classify() node tests
# ---------------------------------------------------------------------------

class TestClassifyNode:
    """Verify classification logic and safe defaults."""

    def _make_clf_mock(self, content: str):
        with patch("main.classifier_llm") as mock_clf:
            result = MagicMock()
            result.content = content
            mock_clf.invoke.return_value = result
            yield mock_clf

    def test_classify_returns_dict(self):
        with patch("main.classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="SIMPLE")
            state: WealthDeskState = {
                "customer_message": "What is the home loan rate?",
                "response": "", "history": [], "query_type": "",
            }
            result = classify(state)
            assert isinstance(result, dict)

    def test_classify_returns_query_type_key(self):
        with patch("main.classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="SIMPLE")
            state: WealthDeskState = {
                "customer_message": "What is the FD rate?",
                "response": "", "history": [], "query_type": "",
            }
            result = classify(state)
            assert "query_type" in result, (
                "classify() must return a dict containing 'query_type'. "
                "Check that your return statement is: return {'query_type': query_type}"
            )

    def test_classify_simple_query(self):
        with patch("main.classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="SIMPLE")
            state: WealthDeskState = {
                "customer_message": "What is the home loan rate?",
                "response": "", "history": [], "query_type": "",
            }
            result = classify(state)
            assert result["query_type"] == "SIMPLE"

    def test_classify_complex_query(self):
        with patch("main.classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="COMPLEX")
            state: WealthDeskState = {
                "customer_message": "Should I take a home loan or invest in FDs?",
                "response": "", "history": [], "query_type": "",
            }
            result = classify(state)
            assert result["query_type"] == "COMPLEX"

    def test_classify_out_of_scope_query(self):
        with patch("main.classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            state: WealthDeskState = {
                "customer_message": "Write me a poem about interest rates.",
                "response": "", "history": [], "query_type": "",
            }
            result = classify(state)
            assert result["query_type"] == "OUT_OF_SCOPE"

    def test_classify_strips_whitespace(self):
        """LLM may return '  SIMPLE  ' with surrounding spaces."""
        with patch("main.classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="  SIMPLE  ")
            state: WealthDeskState = {
                "customer_message": "What is the FD rate?",
                "response": "", "history": [], "query_type": "",
            }
            result = classify(state)
            assert result["query_type"] == "SIMPLE"

    def test_classify_upcases_lowercase_response(self):
        """LLM may return 'simple' in lowercase -- must be uppercased."""
        with patch("main.classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="simple")
            state: WealthDeskState = {
                "customer_message": "What is the FD rate?",
                "response": "", "history": [], "query_type": "",
            }
            result = classify(state)
            assert result["query_type"] == "SIMPLE"

    def test_classify_defaults_to_simple_on_unexpected_output(self):
        """If the LLM outputs something other than the three valid values,
        classify() must default to SIMPLE (fail open)."""
        with patch("main.classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="I am not sure.")
            state: WealthDeskState = {
                "customer_message": "Something ambiguous.",
                "response": "", "history": [], "query_type": "",
            }
            result = classify(state)
            assert result["query_type"] == "SIMPLE", (
                "classify() must default to 'SIMPLE' when the LLM returns an "
                "unexpected value. Check the validation block after parsing the LLM output."
            )

    def test_classify_defaults_to_simple_on_llm_error(self, mock_classifier_error):
        """classify() must not propagate LLM exceptions."""
        state: WealthDeskState = {
            "customer_message": "What is the home loan rate?",
            "response": "", "history": [], "query_type": "",
        }
        result = classify(state)
        assert "query_type" in result
        assert result["query_type"] == "SIMPLE"

    def test_classify_uses_classifier_llm_not_main_llm(self):
        """classify() must call classifier_llm, not the main llm."""
        with patch("main.classifier_llm") as mock_clf, \
             patch("main.llm") as mock_main_llm:
            mock_clf.invoke.return_value = MagicMock(content="SIMPLE")
            state: WealthDeskState = {
                "customer_message": "What is the FD rate?",
                "response": "", "history": [], "query_type": "",
            }
            classify(state)
            mock_clf.invoke.assert_called_once()
            mock_main_llm.invoke.assert_not_called()

    def test_classify_does_not_include_history_in_messages(self):
        """History must NOT be passed to the classifier LLM."""
        from langchain_core.messages import AIMessage, HumanMessage

        with patch("main.classifier_llm") as mock_clf:
            mock_clf.invoke.return_value = MagicMock(content="SIMPLE")
            state: WealthDeskState = {
                "customer_message": "What is the FD rate?",
                "response": "",
                "history": [
                    {"role": "user",      "content": "I earn Rs. 80,000 per month."},
                    {"role": "assistant", "content": "Noted. How can I help?"},
                ],
                "query_type": "",
            }
            classify(state)
            call_args = mock_clf.invoke.call_args
            messages  = call_args[0][0]
            ai_messages = [m for m in messages if isinstance(m, AIMessage)]
            assert len(ai_messages) == 0, (
                "classify() must NOT include history in its LLM call. "
                "Classification should be based on the current question only."
            )


# ---------------------------------------------------------------------------
# route_query() tests
# ---------------------------------------------------------------------------

class TestRouteQuery:
    """route_query() is a pure function -- no LLM, fully deterministic."""

    def test_simple_routes_to_respond(self):
        state: WealthDeskState = {
            "customer_message": "test", "response": "",
            "history": [], "query_type": "SIMPLE",
        }
        assert route_query(state) == "respond"

    def test_complex_routes_to_escalate(self):
        state: WealthDeskState = {
            "customer_message": "test", "response": "",
            "history": [], "query_type": "COMPLEX",
        }
        assert route_query(state) == "escalate"

    def test_out_of_scope_routes_to_decline(self):
        state: WealthDeskState = {
            "customer_message": "test", "response": "",
            "history": [], "query_type": "OUT_OF_SCOPE",
        }
        assert route_query(state) == "decline"

    def test_missing_query_type_defaults_to_respond(self):
        """If query_type is missing from state (e.g. old checkpoint), default to respond."""
        state = {
            "customer_message": "test", "response": "", "history": [],
            # no query_type key
        }
        assert route_query(state) == "respond"

    def test_empty_query_type_defaults_to_respond(self):
        state: WealthDeskState = {
            "customer_message": "test", "response": "",
            "history": [], "query_type": "",
        }
        assert route_query(state) == "respond"

    def test_route_query_returns_string(self):
        state: WealthDeskState = {
            "customer_message": "test", "response": "",
            "history": [], "query_type": "SIMPLE",
        }
        result = route_query(state)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# escalate() and decline() node tests
# ---------------------------------------------------------------------------

class TestEscalateNode:
    """escalate() must not call the LLM and must include RM contact info."""

    def test_escalate_does_not_call_llm(self):
        with patch("main.llm") as mock_llm, \
             patch("main.classifier_llm") as mock_clf:
            state: WealthDeskState = {
                "customer_message": "Should I take a home loan or invest?",
                "response": "", "history": [], "query_type": "COMPLEX",
            }
            escalate(state)
            mock_llm.invoke.assert_not_called()
            mock_clf.invoke.assert_not_called()

    def test_escalate_returns_response_key(self):
        state: WealthDeskState = {
            "customer_message": "Should I take a home loan or invest?",
            "response": "", "history": [], "query_type": "COMPLEX",
        }
        result = escalate(state)
        assert "response" in result

    def test_escalate_response_is_non_empty_string(self):
        state: WealthDeskState = {
            "customer_message": "Should I take a home loan or invest?",
            "response": "", "history": [], "query_type": "COMPLEX",
        }
        result = escalate(state)
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_escalate_response_contains_rm_referral(self):
        """The escalation response must direct the customer to a Relationship Manager."""
        state: WealthDeskState = {
            "customer_message": "Which loan is best for me?",
            "response": "", "history": [], "query_type": "COMPLEX",
        }
        result = escalate(state)
        lower = result["response"].lower()
        has_rm_mention = (
            "relationship manager" in lower
            or "branch" in lower
            or "1800" in result["response"]
        )
        assert has_rm_mention, (
            "escalate() response must reference a Relationship Manager or branch. "
            "Check ESCALATE_RESPONSE in the solution."
        )

    def test_escalate_updates_history(self):
        state: WealthDeskState = {
            "customer_message": "Which loan is best for me?",
            "response": "", "history": [], "query_type": "COMPLEX",
        }
        result = escalate(state)
        assert "history" in result
        assert len(result["history"]) == 2

    def test_escalate_preserves_previous_history(self):
        existing = [
            {"role": "user",      "content": "What is the FD rate?"},
            {"role": "assistant", "content": "6.8% p.a."},
        ]
        state: WealthDeskState = {
            "customer_message": "Which FD is best for retirement?",
            "response": "", "history": existing, "query_type": "COMPLEX",
        }
        result = escalate(state)
        assert len(result["history"]) == 4
        assert result["history"][0] == existing[0]


class TestDeclineNode:
    """decline() must not call the LLM and must be a polite refusal."""

    def test_decline_does_not_call_llm(self):
        with patch("main.llm") as mock_llm, \
             patch("main.classifier_llm") as mock_clf:
            state: WealthDeskState = {
                "customer_message": "Write me a poem.",
                "response": "", "history": [], "query_type": "OUT_OF_SCOPE",
            }
            decline(state)
            mock_llm.invoke.assert_not_called()
            mock_clf.invoke.assert_not_called()

    def test_decline_returns_response_key(self):
        state: WealthDeskState = {
            "customer_message": "Write me a poem.",
            "response": "", "history": [], "query_type": "OUT_OF_SCOPE",
        }
        result = decline(state)
        assert "response" in result
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_decline_response_is_polite(self):
        """The decline must mention what WealthDesk CAN help with, not just refuse."""
        state: WealthDeskState = {
            "customer_message": "What is the weather?",
            "response": "", "history": [], "query_type": "OUT_OF_SCOPE",
        }
        result = decline(state)
        lower = result["response"].lower()
        has_scope_mention = (
            "loan" in lower or "deposit" in lower or "bnb" in lower
        )
        assert has_scope_mention, (
            "A good decline message mentions what the agent CAN help with, "
            "not just what it cannot. Check DECLINE_RESPONSE in the solution."
        )

    def test_decline_updates_history(self):
        state: WealthDeskState = {
            "customer_message": "Write me a poem.",
            "response": "", "history": [], "query_type": "OUT_OF_SCOPE",
        }
        result = decline(state)
        assert "history" in result
        assert len(result["history"]) == 2


# ---------------------------------------------------------------------------
# Full graph routing tests
# ---------------------------------------------------------------------------

class TestGraphRouting:
    """Invoke the full compiled graph and verify the correct path was taken."""

    def test_simple_query_sets_query_type_simple(
        self, mock_llm_simple, memory_checkpointer
    ):
        graph  = build_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "route-simple"}}
        result = graph.invoke(
            {"customer_message": "What is the home loan rate?", "response": ""},
            config=config,
        )
        assert result.get("query_type") == "SIMPLE"

    def test_simple_query_calls_main_llm(self, mock_llm_simple, memory_checkpointer):
        """SIMPLE path: classifier_llm for classify, main llm for respond."""
        main_llm, clf_llm = mock_llm_simple
        graph  = build_graph(checkpointer=memory_checkpointer)
        config = {"configurable": {"thread_id": "route-simple-llm"}}
        graph.invoke(
            {"customer_message": "What is the FD rate?", "response": ""},
            config=config,
        )
        clf_llm.invoke.assert_called_once()
        main_llm.invoke.assert_called_once()

    def test_complex_query_does_not_call_main_llm(
        self, mock_llm_complex, memory_checkpointer
    ):
        """COMPLEX path: classify (classifier_llm) + escalate (no LLM)."""
        with patch("main.llm") as mock_main_llm:
            graph  = build_graph(checkpointer=memory_checkpointer)
            config = {"configurable": {"thread_id": "route-complex"}}
            result = graph.invoke(
                {"customer_message": "Should I take a home loan or invest?", "response": ""},
                config=config,
            )
            mock_main_llm.invoke.assert_not_called()
            assert result.get("query_type") == "COMPLEX"
            assert result["response"] == ESCALATE_RESPONSE

    def test_out_of_scope_query_does_not_call_main_llm(
        self, mock_llm_out_of_scope, memory_checkpointer
    ):
        """OUT_OF_SCOPE path: classify + decline (no LLM)."""
        with patch("main.llm") as mock_main_llm:
            graph  = build_graph(checkpointer=memory_checkpointer)
            config = {"configurable": {"thread_id": "route-oos"}}
            result = graph.invoke(
                {"customer_message": "Write me a poem.", "response": ""},
                config=config,
            )
            mock_main_llm.invoke.assert_not_called()
            assert result.get("query_type") == "OUT_OF_SCOPE"
            assert result["response"] == DECLINE_RESPONSE

    def test_all_paths_return_non_empty_response(
        self, mock_llm_simple, memory_checkpointer
    ):
        """All three routing paths must produce a non-empty response."""
        main_llm, clf_llm = mock_llm_simple
        graph = build_graph(checkpointer=memory_checkpointer)

        for query_type, question in [
            ("SIMPLE",      "What is the FD rate?"),
            ("COMPLEX",     "Which loan should I take?"),
            ("OUT_OF_SCOPE","Write a poem."),
        ]:
            clf_llm.invoke.return_value = MagicMock(content=query_type)
            config = {"configurable": {"thread_id": f"all-paths-{query_type}"}}
            result = graph.invoke(
                {"customer_message": question, "response": ""},
                config=config,
            )
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 0, (
                f"Path {query_type} produced an empty response."
            )

    def test_all_paths_update_history(self, mock_llm_simple, memory_checkpointer):
        """All three routing paths must add two items to history."""
        main_llm, clf_llm = mock_llm_simple
        graph = build_graph(checkpointer=memory_checkpointer)

        for query_type, question in [
            ("SIMPLE",      "What is the FD rate?"),
            ("COMPLEX",     "Which loan should I take?"),
            ("OUT_OF_SCOPE","Write a poem."),
        ]:
            clf_llm.invoke.return_value = MagicMock(content=query_type)
            config = {"configurable": {"thread_id": f"history-{query_type}"}}
            result = graph.invoke(
                {"customer_message": question, "response": ""},
                config=config,
            )
            assert len(result["history"]) == 2, (
                f"Path {query_type} should add 2 items to an empty history. "
                f"Got {len(result['history'])}."
            )


# ---------------------------------------------------------------------------
# Memory + routing combined
# ---------------------------------------------------------------------------

class TestMemoryWithRouting:
    """Session 3 must preserve all of Session 2's memory behaviour."""

    def test_history_accumulates_across_simple_turns(
        self, mock_llm_simple, memory_checkpointer
    ):
        main_llm, clf_llm = mock_llm_simple
        graph     = build_graph(checkpointer=memory_checkpointer)
        thread_id = "mem-simple"
        config    = {"configurable": {"thread_id": thread_id}}

        clf_llm.invoke.return_value = MagicMock(content="SIMPLE")
        graph.invoke(
            {"customer_message": "What is the home loan rate?", "response": ""},
            config=config,
        )

        clf_llm.invoke.return_value = MagicMock(content="SIMPLE")
        result = graph.invoke(
            {"customer_message": "And the personal loan rate?", "response": ""},
            config=config,
        )
        assert len(result["history"]) == 4

    def test_history_accumulates_across_mixed_routes(
        self, mock_llm_simple, memory_checkpointer
    ):
        """SIMPLE turn + COMPLEX turn: history should have 4 items total."""
        main_llm, clf_llm = mock_llm_simple
        graph     = build_graph(checkpointer=memory_checkpointer)
        thread_id = "mem-mixed"
        config    = {"configurable": {"thread_id": thread_id}}

        clf_llm.invoke.return_value = MagicMock(content="SIMPLE")
        graph.invoke(
            {"customer_message": "What is the FD rate?", "response": ""},
            config=config,
        )

        clf_llm.invoke.return_value = MagicMock(content="COMPLEX")
        result = graph.invoke(
            {"customer_message": "Which is better for me, FD or PPF?", "response": ""},
            config=config,
        )
        assert len(result["history"]) == 4

    def test_different_threads_independent(
        self, mock_llm_simple, memory_checkpointer
    ):
        main_llm, clf_llm = mock_llm_simple
        graph = build_graph(checkpointer=memory_checkpointer)

        clf_llm.invoke.return_value = MagicMock(content="SIMPLE")
        graph.invoke(
            {"customer_message": "What is the FD rate?", "response": ""},
            config={"configurable": {"thread_id": "thread-X"}},
        )

        clf_llm.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
        result = graph.invoke(
            {"customer_message": "Write me a poem.", "response": ""},
            config={"configurable": {"thread_id": "thread-Y"}},
        )
        assert len(result["history"]) == 2
