"""
s04/tests/test_s04.py
---------------------
Unit tests for Session 4: ChromaDB RAG + LangSmith Tracing.

Run from the wealthdesk/ directory:
    pytest s04/tests/ -v

All tests run without a live Groq API key, ChromaDB, or HuggingFace model.
The vectorstore and LLMs are mocked throughout.

Design rationale:
  - State tests: catch regressions if retrieved_docs is removed.
  - retrieve_docs() tests: verify correct interaction with vectorstore,
    source attribution format, and graceful handling of None vectorstore
    and exceptions.
  - respond() tests: verify that retrieved docs appear in the system message
    and that the node falls back correctly when docs is empty.
  - Graph routing tests: SIMPLE goes through retrieve_docs; COMPLEX and
    OUT_OF_SCOPE skip it.
  - Memory tests: history accumulates correctly through all paths.
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
    DECLINE_RESPONSE,
    ESCALATE_RESPONSE,
    RETRIEVAL_K,
    WealthDeskState,
    build_graph,
    classify,
    decline,
    escalate,
    respond,
    retrieve_docs,
    route_query,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_doc(page_content: str, source: str = "home_loan_guide.md") -> MagicMock:
    """Create a MagicMock that mimics a LangChain Document."""
    doc = MagicMock()
    doc.page_content = page_content
    doc.metadata = {"source": source}
    return doc


def _mock_vectorstore_with(docs: list) -> MagicMock:
    """Return a mock vectorstore whose similarity_search returns `docs`."""
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = docs
    return mock_vs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def memory_checkpointer():
    return MemorySaver()


@pytest.fixture
def mock_llm_simple():
    """Patch classifier_llm (SIMPLE) and main llm (respond answer)."""
    with patch("main.llm") as mock_main, patch("main.classifier_llm") as mock_clf:
        mock_clf.invoke.return_value = MagicMock(content="SIMPLE")
        mock_main.invoke.return_value = MagicMock(
            content="The BNB home loan requires salary slips and a PAN card. WealthDesk | BNB"
        )
        yield mock_main, mock_clf


@pytest.fixture
def mock_vectorstore():
    """Patch main.vectorstore with a mock that returns one document."""
    doc = _make_mock_doc(
        "For a home loan, BNB requires: last 3 months' salary slips, "
        "PAN card, Aadhaar card, Form 16, and 6 months' bank statements.",
        source="home_loan_guide.md",
    )
    mock_vs = _mock_vectorstore_with([doc])
    with patch("main.vectorstore", mock_vs):
        yield mock_vs


# ---------------------------------------------------------------------------
# State structure tests
# ---------------------------------------------------------------------------

class TestWealthDeskState:

    def test_state_has_customer_message(self):
        assert "customer_message" in WealthDeskState.__annotations__

    def test_state_has_response(self):
        assert "response" in WealthDeskState.__annotations__

    def test_state_has_history(self):
        assert "history" in WealthDeskState.__annotations__

    def test_state_has_query_type(self):
        assert "query_type" in WealthDeskState.__annotations__

    def test_state_has_retrieved_docs(self):
        """Session 4 adds retrieved_docs. This test catches regressions in later sessions."""
        assert "retrieved_docs" in WealthDeskState.__annotations__, (
            "WealthDeskState must have a 'retrieved_docs' field (added in Session 4). "
            "Add it after 'query_type' with type hint list[str]."
        )

    def test_retrieved_docs_is_list_type(self):
        annotation = WealthDeskState.__annotations__["retrieved_docs"]
        origin = getattr(annotation, "__origin__", annotation)
        assert origin is list, (
            f"'retrieved_docs' should be typed as list[str] but found: {annotation}"
        )

    def test_state_instantiable_with_all_fields(self):
        state: WealthDeskState = {
            "customer_message": "What documents do I need?",
            "response":         "",
            "history":          [],
            "query_type":       "SIMPLE",
            "retrieved_docs":   [],
        }
        assert state["retrieved_docs"] == []

    def test_state_retrieved_docs_accepts_strings(self):
        state: WealthDeskState = {
            "customer_message": "test",
            "response":         "",
            "history":          [],
            "query_type":       "SIMPLE",
            "retrieved_docs":   ["[home_loan_guide.md]\nSome policy text."],
        }
        assert len(state["retrieved_docs"]) == 1


# ---------------------------------------------------------------------------
# retrieve_docs() node tests
# ---------------------------------------------------------------------------

class TestRetrieveDocsNode:

    def _state(self, question: str = "What documents do I need for a home loan?") -> WealthDeskState:
        return {
            "customer_message": question,
            "response":         "",
            "history":          [],
            "query_type":       "SIMPLE",
            "retrieved_docs":   [],
        }

    def test_retrieve_docs_returns_dict(self, mock_vectorstore):
        result = retrieve_docs(self._state())
        assert isinstance(result, dict)

    def test_retrieve_docs_returns_retrieved_docs_key(self, mock_vectorstore):
        result = retrieve_docs(self._state())
        assert "retrieved_docs" in result, (
            "retrieve_docs() must return a dict with 'retrieved_docs' key. "
            "Check your return statement."
        )

    def test_retrieve_docs_returns_list(self, mock_vectorstore):
        result = retrieve_docs(self._state())
        assert isinstance(result["retrieved_docs"], list)

    def test_retrieve_docs_calls_similarity_search(self, mock_vectorstore):
        retrieve_docs(self._state("What documents do I need?"))
        mock_vectorstore.similarity_search.assert_called_once()

    def test_retrieve_docs_passes_question_to_search(self, mock_vectorstore):
        question = "What documents do I need for a home loan?"
        retrieve_docs(self._state(question))
        call_args = mock_vectorstore.similarity_search.call_args
        assert call_args[0][0] == question or call_args[1].get("query") == question, (
            "retrieve_docs() must pass state['customer_message'] as the search query."
        )

    def test_retrieve_docs_passes_k_parameter(self, mock_vectorstore):
        retrieve_docs(self._state())
        call_args = mock_vectorstore.similarity_search.call_args
        k_value   = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("k")
        assert k_value == RETRIEVAL_K, (
            f"retrieve_docs() must pass k={RETRIEVAL_K} to similarity_search. "
            f"Got k={k_value}."
        )

    def test_retrieve_docs_formats_source_in_output(self, mock_vectorstore):
        """Each retrieved string must contain the source filename."""
        result = retrieve_docs(self._state())
        assert len(result["retrieved_docs"]) > 0
        first = result["retrieved_docs"][0]
        assert "home_loan_guide.md" in first, (
            "retrieve_docs() must include the source filename in each retrieved string. "
            "Format: '[filename.md]\\n<chunk text>'"
        )

    def test_retrieve_docs_includes_page_content(self, mock_vectorstore):
        """The chunk text must appear in the output."""
        result = retrieve_docs(self._state())
        combined = " ".join(result["retrieved_docs"])
        assert "salary slips" in combined or "home loan" in combined.lower()

    def test_retrieve_docs_returns_empty_when_vectorstore_is_none(self):
        """If vectorstore is None, retrieve_docs() must return an empty list."""
        with patch("main.vectorstore", None):
            result = retrieve_docs(self._state())
        assert result == {"retrieved_docs": []}, (
            "retrieve_docs() must return {'retrieved_docs': []} when vectorstore is None. "
            "Check the None guard at the top of the function."
        )

    def test_retrieve_docs_returns_empty_on_exception(self):
        """A vectorstore error must be caught and return an empty list."""
        mock_vs = MagicMock()
        mock_vs.similarity_search.side_effect = Exception("ChromaDB connection error")
        with patch("main.vectorstore", mock_vs):
            result = retrieve_docs(self._state())
        assert result == {"retrieved_docs": []}, (
            "retrieve_docs() must catch exceptions from similarity_search "
            "and return {'retrieved_docs': []} instead of propagating the error."
        )

    def test_retrieve_docs_does_not_call_llm(self, mock_vectorstore):
        """RAG retrieval must not call any LLM."""
        with patch("main.llm") as mock_llm, \
             patch("main.classifier_llm") as mock_clf:
            retrieve_docs(self._state())
            mock_llm.invoke.assert_not_called()
            mock_clf.invoke.assert_not_called()


# ---------------------------------------------------------------------------
# respond() node tests -- context injection
# ---------------------------------------------------------------------------

class TestRespondWithContext:

    def _state_with_docs(self, docs: list[str]) -> WealthDeskState:
        return {
            "customer_message": "What documents do I need for a home loan?",
            "response":         "",
            "history":          [],
            "query_type":       "SIMPLE",
            "retrieved_docs":   docs,
        }

    def test_respond_includes_retrieved_docs_in_system_message(self):
        """When retrieved_docs is non-empty, the docs must appear in the LLM call."""
        chunk = "[home_loan_guide.md]\nRequires: salary slips, PAN, Aadhaar."
        state = self._state_with_docs([chunk])

        with patch("main.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Docs needed: salary slips.")
            respond(state)

        call_args    = mock_llm.invoke.call_args
        messages     = call_args[0][0]
        system_text  = messages[0].content
        assert "salary slips" in system_text or "home_loan_guide.md" in system_text, (
            "respond() must include retrieved_docs content in the system message "
            "when retrieved_docs is non-empty. Check that you are building "
            "'system_content' from SYSTEM_PROMPT + context_block."
        )

    def test_respond_without_docs_uses_base_system_prompt(self):
        """When retrieved_docs is empty, the system message must be the base SYSTEM_PROMPT."""
        from main import SYSTEM_PROMPT

        state = self._state_with_docs([])

        with patch("main.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Standard answer.")
            respond(state)

        call_args   = mock_llm.invoke.call_args
        messages    = call_args[0][0]
        system_text = messages[0].content
        assert system_text == SYSTEM_PROMPT, (
            "When retrieved_docs is empty, respond() must use SYSTEM_PROMPT unchanged. "
            "Check the if/else block that builds system_content."
        )

    def test_respond_context_contains_policy_keyword(self):
        """The policy keyword from the retrieved chunk must reach the LLM."""
        chunk = "[home_loan_guide.md]\nMinimum age for home loan applicant is 21 years."
        state = self._state_with_docs([chunk])

        with patch("main.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Min age is 21.")
            respond(state)

        call_args   = mock_llm.invoke.call_args
        messages    = call_args[0][0]
        system_text = messages[0].content
        assert "21" in system_text, (
            "The retrieved chunk content must appear in the system message "
            "sent to the LLM."
        )

    def test_respond_with_multiple_docs(self):
        """Multiple retrieved chunks must all appear in the system message."""
        chunks = [
            "[home_loan_guide.md]\nSalary slips required.",
            "[bnb_policy.md]\nPAN card mandatory for all loans.",
        ]
        state = self._state_with_docs(chunks)

        with patch("main.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Both docs needed.")
            respond(state)

        call_args   = mock_llm.invoke.call_args
        messages    = call_args[0][0]
        system_text = messages[0].content
        assert "salary slips" in system_text.lower() or "Salary" in system_text
        assert "PAN" in system_text

    def test_respond_updates_history_with_docs(self):
        """History must be updated even when retrieved_docs is non-empty."""
        chunk = "[faq.md]\nFD can be opened online."
        state = self._state_with_docs([chunk])

        with patch("main.llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="Yes, FD can be opened online.")
            result = respond(state)

        assert len(result["history"]) == 2


# ---------------------------------------------------------------------------
# route_query() tests
# ---------------------------------------------------------------------------

class TestRouteQuery:

    def _state(self, query_type: str) -> dict:
        return {"customer_message": "test", "response": "",
                "history": [], "query_type": query_type, "retrieved_docs": []}

    def test_simple_routes_to_retrieve_docs(self):
        """Session 4: SIMPLE must route to 'retrieve_docs', not 'respond'."""
        assert route_query(self._state("SIMPLE")) == "retrieve_docs", (
            "In Session 4, route_query() must return 'retrieve_docs' for SIMPLE queries. "
            "Check TODO 4a in the starter -- the return value changed from 'respond'."
        )

    def test_complex_routes_to_escalate(self):
        assert route_query(self._state("COMPLEX")) == "escalate"

    def test_out_of_scope_routes_to_decline(self):
        assert route_query(self._state("OUT_OF_SCOPE")) == "decline"

    def test_default_routes_to_retrieve_docs(self):
        state = {"customer_message": "test", "response": "", "history": [], "retrieved_docs": []}
        assert route_query(state) == "retrieve_docs"


# ---------------------------------------------------------------------------
# Graph routing tests
# ---------------------------------------------------------------------------

class TestGraphRouting:

    def test_simple_path_calls_vectorstore(self, mock_vectorstore, mock_llm_simple, memory_checkpointer):
        """SIMPLE path: vectorstore must be queried."""
        with patch("main.vectorstore", mock_vectorstore):
            graph  = build_graph(checkpointer=memory_checkpointer)
            config = {"configurable": {"thread_id": "route-simple-rag"}}
            graph.invoke(
                {"customer_message": "What documents do I need for a home loan?", "response": ""},
                config=config,
            )
        mock_vectorstore.similarity_search.assert_called_once()

    def test_simple_path_sets_retrieved_docs_in_result(self, mock_vectorstore, mock_llm_simple, memory_checkpointer):
        """After a SIMPLE invocation, result must contain retrieved_docs."""
        with patch("main.vectorstore", mock_vectorstore):
            graph  = build_graph(checkpointer=memory_checkpointer)
            config = {"configurable": {"thread_id": "route-simple-docs"}}
            result = graph.invoke(
                {"customer_message": "What documents do I need?", "response": ""},
                config=config,
            )
        assert "retrieved_docs" in result
        assert isinstance(result["retrieved_docs"], list)
        assert len(result["retrieved_docs"]) > 0

    def test_complex_path_skips_vectorstore(self, memory_checkpointer):
        """COMPLEX path must NOT query the vectorstore."""
        mock_vs = MagicMock()
        with patch("main.vectorstore", mock_vs), \
             patch("main.classifier_llm") as mock_clf, \
             patch("main.llm"):
            mock_clf.invoke.return_value = MagicMock(content="COMPLEX")
            graph  = build_graph(checkpointer=memory_checkpointer)
            config = {"configurable": {"thread_id": "route-complex-no-rag"}}
            result = graph.invoke(
                {"customer_message": "Should I invest or take a home loan?", "response": ""},
                config=config,
            )
        mock_vs.similarity_search.assert_not_called()
        assert result["response"] == ESCALATE_RESPONSE

    def test_out_of_scope_path_skips_vectorstore(self, memory_checkpointer):
        """OUT_OF_SCOPE path must NOT query the vectorstore."""
        mock_vs = MagicMock()
        with patch("main.vectorstore", mock_vs), \
             patch("main.classifier_llm") as mock_clf, \
             patch("main.llm"):
            mock_clf.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph  = build_graph(checkpointer=memory_checkpointer)
            config = {"configurable": {"thread_id": "route-oos-no-rag"}}
            graph.invoke(
                {"customer_message": "Write me a poem.", "response": ""},
                config=config,
            )
        mock_vs.similarity_search.assert_not_called()

    def test_all_paths_produce_non_empty_response(self, mock_vectorstore, memory_checkpointer):
        for query_type, question in [
            ("SIMPLE",       "What documents do I need?"),
            ("COMPLEX",      "Which loan should I take?"),
            ("OUT_OF_SCOPE", "Write a poem."),
        ]:
            with patch("main.vectorstore", mock_vectorstore), \
                 patch("main.classifier_llm") as mock_clf, \
                 patch("main.llm") as mock_llm:
                mock_clf.invoke.return_value = MagicMock(content=query_type)
                mock_llm.invoke.return_value = MagicMock(content="Some answer.")
                graph  = build_graph(checkpointer=MemorySaver())
                config = {"configurable": {"thread_id": f"all-paths-{query_type}"}}
                result = graph.invoke(
                    {"customer_message": question, "response": ""},
                    config=config,
                )
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 0


# ---------------------------------------------------------------------------
# Memory tests
# ---------------------------------------------------------------------------

class TestMemoryWithRAG:

    def test_history_accumulates_across_simple_turns(self, mock_vectorstore, mock_llm_simple, memory_checkpointer):
        with patch("main.vectorstore", mock_vectorstore):
            graph     = build_graph(checkpointer=memory_checkpointer)
            thread_id = "mem-rag-simple"
            config    = {"configurable": {"thread_id": thread_id}}

            graph.invoke(
                {"customer_message": "What documents do I need?", "response": ""},
                config=config,
            )
            result = graph.invoke(
                {"customer_message": "And for a personal loan?", "response": ""},
                config=config,
            )
        assert len(result["history"]) == 4

    def test_different_threads_isolated(self, mock_vectorstore, mock_llm_simple, memory_checkpointer):
        with patch("main.vectorstore", mock_vectorstore):
            graph = build_graph(checkpointer=memory_checkpointer)

            graph.invoke(
                {"customer_message": "What is the FD rate?", "response": ""},
                config={"configurable": {"thread_id": "rag-thread-X"}},
            )
            result = graph.invoke(
                {"customer_message": "What documents do I need?", "response": ""},
                config={"configurable": {"thread_id": "rag-thread-Y"}},
            )
        assert len(result["history"]) == 2
