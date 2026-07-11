"""
Session 3 tests for query routing and conditional edges.
Run from the wealthdesk/ directory:
    pytest s03/tests/ -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SOLUTION_DIR = Path(__file__).parent.parent / "starter"
for _k in list(sys.modules):
    if _k == "main" or _k.startswith("main."):
        sys.modules.pop(_k)
sys.path.insert(0, str(SOLUTION_DIR))

import main as solution  # noqa: E402
from main import (
    WealthDeskState,
    classify,
    route_query,
    build_graph,
    ESCALATE_RESPONSE,
    DECLINE_RESPONSE,
)  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_classifier_response():
    with patch.object(solution, "classifier_llm") as mock_llm:
        mock_result = MagicMock()
        mock_result.content = "COMPLEX"
        mock_llm.invoke.return_value = mock_result
        yield mock_llm


@pytest.fixture
def mock_classifier_error():
    with patch.object(solution, "classifier_llm") as mock_llm:
        mock_llm.invoke.side_effect = Exception("Classifier timeout")
        yield mock_llm


@pytest.fixture
def mock_llm_response():
    with patch.object(solution, "llm") as mock_llm:
        mock_result = MagicMock()
        mock_result.content = "The BNB home loan rate is 8.5% per annum. WealthDesk | Bharat National Bank"
        mock_llm.invoke.return_value = mock_result
        yield mock_llm


# ---------------------------------------------------------------------------
# State structure tests
# ---------------------------------------------------------------------------

class TestWealthDeskState:
    def test_state_has_query_type_field(self):
        assert "query_type" in WealthDeskState.__annotations__

    def test_query_type_is_str(self):
        assert WealthDeskState.__annotations__["query_type"] is str

    def test_state_can_include_all_fields(self):
        state: WealthDeskState = {
            "customer_message": "What is the home loan rate?",
            "response": "",
            "history": [],
            "query_type": "SIMPLE",
        }
        assert state["query_type"] == "SIMPLE"


# ---------------------------------------------------------------------------
# classify() node tests
# ---------------------------------------------------------------------------

class TestClassifyNode:
    def test_classify_returns_query_type(self, mock_classifier_response):
        state: WealthDeskState = {
            "customer_message": "Should I take a home loan or use my savings?",
            "response": "",
            "history": [],
            "query_type": "",
        }
        result = classify(state)
        assert isinstance(result, dict)
        assert result["query_type"] == "COMPLEX"

    def test_classify_calls_classifier_llm_with_system_and_human_messages(self, mock_classifier_response):
        from langchain_core.messages import HumanMessage, SystemMessage

        customer_question = "What documents do I need for an FD?"
        state: WealthDeskState = {
            "customer_message": customer_question,
            "response": "",
            "history": [],
            "query_type": "",
        }
        classify(state)

        call_args = mock_classifier_response.invoke.call_args
        messages = call_args[0][0]
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        human_messages = [m for m in messages if isinstance(m, HumanMessage)]

        assert len(system_messages) == 1
        assert len(human_messages) == 1
        assert human_messages[0].content == customer_question

    def test_classify_defaults_to_simple_on_unexpected_output(self, mock_classifier_response):
        mock_classifier_response.invoke.return_value.content = "Maybe"
        state: WealthDeskState = {
            "customer_message": "Tell me a poem.",
            "response": "",
            "history": [],
            "query_type": "",
        }
        result = classify(state)
        assert result["query_type"] == "SIMPLE"

    def test_classify_defaults_to_simple_on_error(self, mock_classifier_error):
        state: WealthDeskState = {
            "customer_message": "What is the stock market doing today?",
            "response": "",
            "history": [],
            "query_type": "",
        }
        result = classify(state)
        assert result["query_type"] == "SIMPLE"


# ---------------------------------------------------------------------------
# route_query() tests
# ---------------------------------------------------------------------------

class TestRouteQuery:
    def test_route_query_returns_escalate_for_complex(self):
        state: WealthDeskState = {
            "customer_message": "Which FD tenure is best for retirement?",
            "response": "",
            "history": [],
            "query_type": "COMPLEX",
        }
        assert route_query(state) == "escalate"

    def test_route_query_returns_decline_for_out_of_scope(self):
        state: WealthDeskState = {
            "customer_message": "Write me a poem about rain.",
            "response": "",
            "history": [],
            "query_type": "OUT_OF_SCOPE",
        }
        assert route_query(state) == "decline"

    def test_route_query_returns_respond_for_simple(self):
        state: WealthDeskState = {
            "customer_message": "What is the home loan rate?",
            "response": "",
            "history": [],
            "query_type": "SIMPLE",
        }
        assert route_query(state) == "respond"

    def test_route_query_defaults_to_respond_if_missing_query_type(self):
        state = {
            "customer_message": "What is the car loan rate?",
            "response": "",
            "history": [],
        }
        assert route_query(state) == "respond"


# ---------------------------------------------------------------------------
# Graph routing tests
# ---------------------------------------------------------------------------

class TestGraphRouting:
    def test_build_graph_routes_complex_queries_to_escalate(self, mock_classifier_response, mock_llm_response):
        mock_classifier_response.invoke.return_value.content = "COMPLEX"
        graph = build_graph()
        config = {"configurable": {"thread_id": "test-complex-routing"}}

        result = graph.invoke(
            {
                "customer_message": "Should I take a home loan or use my savings?",
                "response": "",
                "history": [],
            },
            config=config,
        )

        assert result["response"] == ESCALATE_RESPONSE
        assert result["query_type"] == "COMPLEX"

    def test_build_graph_routes_out_of_scope_queries_to_decline(self, mock_classifier_response, mock_llm_response):
        mock_classifier_response.invoke.return_value.content = "OUT_OF_SCOPE"
        graph = build_graph()
        config = {"configurable": {"thread_id": "test-out-of-scope-routing"}}

        result = graph.invoke(
            {
                "customer_message": "Compare BNB with HDFC Bank.",
                "response": "",
                "history": [],
            },
            config=config,
        )

        assert result["response"] == DECLINE_RESPONSE
        assert result["query_type"] == "OUT_OF_SCOPE"

    def test_build_graph_routes_simple_queries_to_respond(self, mock_classifier_response, mock_llm_response):
        mock_classifier_response.invoke.return_value.content = "SIMPLE"
        mock_llm_response.invoke.return_value.content = "BNB fixed deposits start at 6.8% p.a."
        graph = build_graph()
        config = {"configurable": {"thread_id": "test-simple-routing"}}

        result = graph.invoke(
            {
                "customer_message": "What is the FD rate for 1 year?",
                "response": "",
                "history": [],
            },
            config=config,
        )

        assert result["response"] == "BNB fixed deposits start at 6.8% p.a."
        assert result["query_type"] == "SIMPLE"
