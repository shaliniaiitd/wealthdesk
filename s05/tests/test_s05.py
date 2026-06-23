"""
s05/tests/test_s05.py
---------------------
Tests for Session 5: SQLite tool calls.

Run with:
    pytest s05/tests/ -v

Do NOT run alongside other session tests -- all sessions define a module
named 'main' and pytest's module cache causes cross-session patching conflicts.

Test groups:
  TestWealthDeskState    -- state TypedDict has all five fields (unchanged from S04)
  TestQueryRatesTool     -- query_rates() SQL correctness, filtering, output format
  TestQueryBranchTool    -- query_branch() SQL correctness, city filter, parameterisation
  TestToolSQLSafety      -- SQL injection protection and parameterised-query enforcement
  TestToolsBinding       -- llm_with_tools exists; tools are @tool decorated; prompt updated
  TestRunToolDispatch    -- _run_tool dispatches correctly; handles unknown names
  TestRespondWithTools   -- respond() calls llm_with_tools; executes tool calls; calls llm again
  TestGraphRouting       -- SIMPLE goes through retrieve_docs -> respond; COMPLEX/OOS skip tools
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

sys.modules.pop("main", None)
SOLUTION_DIR = Path(__file__).parent.parent / "solution"
sys.path.insert(0, str(SOLUTION_DIR))

import main  # noqa: E402
from main import (  # noqa: E402
    WealthDeskState,
    _run_tool,
    build_graph,
    query_branch,
    query_rates,
    respond,
)


# ---------------------------------------------------------------------------
# TestWealthDeskState
# ---------------------------------------------------------------------------

class TestWealthDeskState:
    def test_state_has_customer_message_field(self):
        assert "customer_message" in WealthDeskState.__annotations__

    def test_state_has_response_field(self):
        assert "response" in WealthDeskState.__annotations__

    def test_state_has_history_field(self):
        assert "history" in WealthDeskState.__annotations__

    def test_state_has_query_type_field(self):
        assert "query_type" in WealthDeskState.__annotations__

    def test_state_has_retrieved_docs_field(self):
        assert "retrieved_docs" in WealthDeskState.__annotations__

    def test_state_has_exactly_five_fields(self):
        assert len(WealthDeskState.__annotations__) == 5


# ---------------------------------------------------------------------------
# TestQueryRatesTool
# ---------------------------------------------------------------------------

class TestQueryRatesTool:
    def test_query_rates_loan_returns_home_loan(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = query_rates.invoke({"product_type": "loan"})
        assert "Home Loan" in result

    def test_query_rates_loan_includes_rate(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = query_rates.invoke({"product_type": "loan"})
        assert "8.5" in result

    def test_query_rates_loan_includes_tenure(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = query_rates.invoke({"product_type": "loan"})
        assert "5" in result and "30" in result

    def test_query_rates_fd_returns_fd_data(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = query_rates.invoke({"product_type": "fd"})
        assert "FD" in result
        assert "6.8" in result or "7.1" in result

    def test_query_rates_fd_shows_senior_rate(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = query_rates.invoke({"product_type": "fd"})
        # 6.8 + 0.5 = 7.3 for the 1-year FD senior rate
        assert "7.3" in result

    def test_query_rates_all_contains_loans_and_fds(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = query_rates.invoke({"product_type": "all"})
        assert "Home Loan" in result
        assert "FD" in result

    def test_query_rates_default_is_all(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = query_rates.invoke({})
        assert "Home Loan" in result
        assert "FD" in result

    def test_query_rates_returns_string(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        assert isinstance(query_rates.invoke({"product_type": "loan"}), str)

    def test_query_rates_loan_does_not_include_fd(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = query_rates.invoke({"product_type": "loan"})
        assert "FD" not in result

    def test_query_rates_fd_does_not_include_loan_names(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = query_rates.invoke({"product_type": "fd"})
        assert "Home Loan" not in result


# ---------------------------------------------------------------------------
# TestQueryBranchTool
# ---------------------------------------------------------------------------

class TestQueryBranchTool:
    def test_query_branch_all_returns_multiple(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = query_branch.invoke({"city": "all"})
        assert "Bengaluru" in result
        assert "Mumbai" in result

    def test_query_branch_city_filter_returns_matching(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = query_branch.invoke({"city": "Mumbai"})
        assert "Mumbai" in result
        assert "Andheri" in result

    def test_query_branch_city_filter_excludes_others(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = query_branch.invoke({"city": "Mumbai"})
        assert "Bengaluru" not in result

    def test_query_branch_includes_ifsc(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = query_branch.invoke({"city": "Bengaluru"})
        assert "BNBI" in result

    def test_query_branch_includes_phone(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = query_branch.invoke({"city": "Bengaluru"})
        assert "080" in result

    def test_query_branch_no_match_returns_message(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = query_branch.invoke({"city": "Atlantis"})
        assert "No BNB branches found" in result

    def test_query_branch_default_is_all(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = query_branch.invoke({})
        assert "Bengaluru" in result
        assert "Mumbai" in result

    def test_query_branch_returns_string(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        assert isinstance(query_branch.invoke({"city": "all"}), str)


# ---------------------------------------------------------------------------
# TestToolSQLSafety
# ---------------------------------------------------------------------------

class TestToolSQLSafety:
    def test_query_branch_sql_injection_safe(self, seeded_db, monkeypatch):
        """A city with SQL metacharacters must not crash or corrupt the database."""
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = query_branch.invoke({"city": "'; DROP TABLE branches; --"})
        assert isinstance(result, str)
        # After the injection attempt, normal queries must still work
        normal = query_branch.invoke({"city": "Mumbai"})
        assert "Mumbai" in normal

    def test_query_branch_uses_question_mark_placeholder(self):
        """The WHERE city LIKE clause must use ? not f-string interpolation in SQL."""
        import inspect
        # @tool wraps the original function in StructuredTool; .func is the original
        source = inspect.getsource(query_branch.func)
        assert "LIKE ?" in source, (
            "query_branch must use a ? placeholder for the city parameter. "
            "Never interpolate user input directly into a SQL string."
        )


# ---------------------------------------------------------------------------
# TestToolsBinding
# ---------------------------------------------------------------------------

class TestToolsBinding:
    def test_llm_with_tools_exists(self):
        assert hasattr(main, "llm_with_tools"), (
            "llm_with_tools not found. Create it with llm.bind_tools([query_rates, query_branch])."
        )

    def test_query_rates_is_tool_decorated(self):
        assert hasattr(query_rates, "name"), (
            "query_rates does not appear to be decorated with @tool. "
            "Tool-decorated functions have a .name attribute."
        )

    def test_query_branch_is_tool_decorated(self):
        assert hasattr(query_branch, "name"), (
            "query_branch does not appear to be decorated with @tool."
        )

    def test_query_rates_tool_name(self):
        assert query_rates.name == "query_rates"

    def test_query_branch_tool_name(self):
        assert query_branch.name == "query_branch"

    def test_system_prompt_has_no_hardcoded_rates(self):
        assert "8.5%" not in main.SYSTEM_PROMPT, (
            "Session 5 removes the hardcoded rate table from SYSTEM_PROMPT. "
            "Rates now come from query_rates(). Remove the 'Product reference' block."
        )

    def test_system_prompt_mentions_tools(self):
        prompt_lower = main.SYSTEM_PROMPT.lower()
        assert "tool" in prompt_lower or "database" in prompt_lower, (
            "SYSTEM_PROMPT should instruct the LLM to use database tools for rates. "
            "Check that Rule 3 references tools or the database."
        )


# ---------------------------------------------------------------------------
# TestRunToolDispatch
# ---------------------------------------------------------------------------

class TestRunToolDispatch:
    def test_run_tool_dispatches_query_rates(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = _run_tool("query_rates", {"product_type": "loan"})
        assert "Home Loan" in result, f"Expected loan data, got: {result}"

    def test_run_tool_dispatches_query_branch(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = _run_tool("query_branch", {"city": "Mumbai"})
        assert "Mumbai" in result, f"Expected branch data, got: {result}"

    def test_run_tool_unknown_name_returns_error_string(self):
        result = _run_tool("nonexistent_tool", {})
        assert "Unknown tool" in result
        assert "nonexistent_tool" in result

    def test_run_tool_returns_string(self, seeded_db, monkeypatch):
        monkeypatch.setattr("main.DB_PATH", seeded_db)
        result = _run_tool("query_rates", {"product_type": "fd"})
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# TestRespondWithTools
# ---------------------------------------------------------------------------

class TestRespondWithTools:
    """Tests for respond() tool-calling behaviour.

    These tests mock both llm_with_tools and llm to verify the two-call pattern
    without making real API calls. The vectorstore is also mocked.
    """

    def _make_tool_call_result(self, tool_name, args, call_id="call_abc123"):
        """Build a mock LLM result that requests a tool call."""
        result = MagicMock()
        result.content = ""
        result.tool_calls = [{"id": call_id, "name": tool_name, "args": args}]
        return result

    def _make_text_result(self, content):
        """Build a mock LLM result with a plain text response."""
        result = MagicMock()
        result.content = content
        result.tool_calls = []
        return result

    def test_respond_calls_llm_with_tools_first(self):
        """respond() must use llm_with_tools (not llm) for the first LLM call."""
        state = {
            "customer_message": "What is the home loan rate?",
            "response": "",
            "history": [],
            "query_type": "SIMPLE",
            "retrieved_docs": [],
        }
        with patch("main.llm_with_tools") as mock_wt, \
             patch("main.llm") as mock_llm, \
             patch("main.vectorstore", MagicMock()):
            mock_wt.invoke.return_value = self._make_text_result("The rate is 8.5%.")
            respond(state)
        mock_wt.invoke.assert_called_once()
        mock_llm.invoke.assert_not_called()

    def test_respond_no_tool_calls_returns_first_result(self):
        """When the LLM responds directly (no tool_calls), that content is returned."""
        state = {
            "customer_message": "What are BNB's loan products?",
            "response": "",
            "history": [],
            "query_type": "SIMPLE",
            "retrieved_docs": [],
        }
        expected = "BNB offers home loans, personal loans, and more."
        with patch("main.llm_with_tools") as mock_wt, \
             patch("main.llm"), \
             patch("main.vectorstore", MagicMock()):
            mock_wt.invoke.return_value = self._make_text_result(expected)
            result = respond(state)
        assert result["response"] == expected

    def test_respond_makes_second_call_when_tool_requested(self):
        """When tool_calls are present, respond() must call llm (not llm_with_tools) again."""
        state = {
            "customer_message": "What is the home loan rate?",
            "response": "",
            "history": [],
            "query_type": "SIMPLE",
            "retrieved_docs": [],
        }
        with patch("main.llm_with_tools") as mock_wt, \
             patch("main.llm") as mock_llm, \
             patch("main._run_tool", return_value="Home Loan: 8.5% p.a."), \
             patch("main.vectorstore", MagicMock()):
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_rates", {"product_type": "loan"}
            )
            mock_llm.invoke.return_value = self._make_text_result(
                "The home loan rate is 8.5% p.a. WealthDesk | BNB"
            )
            respond(state)
        mock_llm.invoke.assert_called_once()

    def test_respond_executes_tool_via_run_tool(self):
        """When tool_calls are present, _run_tool must be called with the tool name and args."""
        state = {
            "customer_message": "Where are your Mumbai branches?",
            "response": "",
            "history": [],
            "query_type": "SIMPLE",
            "retrieved_docs": [],
        }
        with patch("main.llm_with_tools") as mock_wt, \
             patch("main.llm") as mock_llm, \
             patch("main._run_tool", return_value="BNB Andheri West...") as mock_rt, \
             patch("main.vectorstore", MagicMock()):
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_branch", {"city": "Mumbai"}
            )
            mock_llm.invoke.return_value = self._make_text_result("Here are the Mumbai branches.")
            respond(state)
        mock_rt.assert_called_once_with("query_branch", {"city": "Mumbai"})

    def test_respond_uses_second_call_content_as_response(self):
        """The final response must come from the second LLM call, not the first."""
        state = {
            "customer_message": "What is the car loan rate?",
            "response": "",
            "history": [],
            "query_type": "SIMPLE",
            "retrieved_docs": [],
        }
        final_answer = "The car loan rate is 9.5% p.a. WealthDesk | BNB"
        with patch("main.llm_with_tools") as mock_wt, \
             patch("main.llm") as mock_llm, \
             patch("main._run_tool", return_value="Car Loan: 9.5% p.a."), \
             patch("main.vectorstore", MagicMock()):
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_rates", {"product_type": "loan"}
            )
            mock_llm.invoke.return_value = self._make_text_result(final_answer)
            result = respond(state)
        assert result["response"] == final_answer

    def test_respond_history_grows_by_two(self):
        """Each turn adds one user entry and one assistant entry to history."""
        state = {
            "customer_message": "What is the FD rate?",
            "response": "",
            "history": [],
            "query_type": "SIMPLE",
            "retrieved_docs": [],
        }
        with patch("main.llm_with_tools") as mock_wt, \
             patch("main.llm"), \
             patch("main.vectorstore", MagicMock()):
            mock_wt.invoke.return_value = self._make_text_result("FD rates start at 6.8%.")
            result = respond(state)
        assert len(result["history"]) == 2
        assert result["history"][0]["role"] == "user"
        assert result["history"][1]["role"] == "assistant"

    def test_respond_appends_tool_message_to_conversation(self):
        """ToolMessage must appear in the messages list passed to the second LLM call."""
        state = {
            "customer_message": "Home loan rate?",
            "response": "",
            "history": [],
            "query_type": "SIMPLE",
            "retrieved_docs": [],
        }
        captured_messages = []

        def capture_invoke(msgs):
            captured_messages.extend(msgs)
            return MagicMock(content="8.5% p.a. WealthDesk | BNB", tool_calls=[])

        with patch("main.llm_with_tools") as mock_wt, \
             patch("main.llm") as mock_llm, \
             patch("main._run_tool", return_value="Home Loan: 8.5% p.a."), \
             patch("main.vectorstore", MagicMock()):
            mock_wt.invoke.return_value = self._make_tool_call_result(
                "query_rates", {"product_type": "loan"}
            )
            mock_llm.invoke.side_effect = capture_invoke
            respond(state)

        from langchain_core.messages import ToolMessage as TM
        tool_messages = [m for m in captured_messages if isinstance(m, TM)]
        assert len(tool_messages) == 1
        assert "Home Loan" in tool_messages[0].content


# ---------------------------------------------------------------------------
# TestGraphRouting
# ---------------------------------------------------------------------------

class TestGraphRouting:
    """Verify graph topology and routing paths using a MemorySaver checkpointer."""

    def _mock_vectorstore(self):
        vs = MagicMock()
        vs.similarity_search.return_value = []
        return vs

    def test_simple_path_calls_llm_with_tools(self):
        """A SIMPLE query must go through respond(), which uses llm_with_tools."""
        from langgraph.checkpoint.memory import MemorySaver

        with patch("main.llm_with_tools") as mock_wt, \
             patch("main.classifier_llm") as mock_cl, \
             patch("main.llm"), \
             patch("main.vectorstore", self._mock_vectorstore()):
            mock_cl.invoke.return_value = MagicMock(content="SIMPLE")
            mock_wt.invoke.return_value = MagicMock(
                content="The home loan rate is 8.5%.", tool_calls=[]
            )
            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "test-simple"}}
            graph.invoke(
                {"customer_message": "What is the home loan rate?", "response": ""},
                config=config,
            )
        mock_wt.invoke.assert_called_once()

    def test_complex_path_skips_llm_with_tools(self):
        """A COMPLEX query routes to escalate -- llm_with_tools must not be called."""
        from langgraph.checkpoint.memory import MemorySaver

        with patch("main.llm_with_tools") as mock_wt, \
             patch("main.classifier_llm") as mock_cl, \
             patch("main.vectorstore", self._mock_vectorstore()):
            mock_cl.invoke.return_value = MagicMock(content="COMPLEX")
            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "test-complex"}}
            result = graph.invoke(
                {"customer_message": "Should I invest in FD or pay off my loan?", "response": ""},
                config=config,
            )
        mock_wt.invoke.assert_not_called()
        assert "Relationship Manager" in result["response"]

    def test_out_of_scope_path_skips_llm_with_tools(self):
        """An OUT_OF_SCOPE query routes to decline -- llm_with_tools must not be called."""
        from langgraph.checkpoint.memory import MemorySaver

        with patch("main.llm_with_tools") as mock_wt, \
             patch("main.classifier_llm") as mock_cl, \
             patch("main.vectorstore", self._mock_vectorstore()):
            mock_cl.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph = build_graph(checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": "test-oos"}}
            result = graph.invoke(
                {"customer_message": "What is the weather today?", "response": ""},
                config=config,
            )
        mock_wt.invoke.assert_not_called()
        assert "only help with BNB" in result["response"]
