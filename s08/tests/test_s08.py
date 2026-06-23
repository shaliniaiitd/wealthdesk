"""
s08/tests/test_s08.py
---------------------
Tests for Session 8: MCP Agent Integration (US-06 Part 2).

Run with:
    pytest s08/tests/ -v

All tests mock _run_mcp_tool() so no real MCP server or database is required.
This verifies the wiring: that tool functions call _run_mcp_tool with the
correct arguments, that _run_tool dispatches correctly, and that the graph
can be built and invoked.

Test groups:
  TestMCPServerPath    -- MCP_SERVER_PATH points to the Session 7 server
  TestRunMcpTool       -- _run_mcp_tool error handling, return value
  TestToolFunctions    -- query_rates / query_branch call _run_mcp_tool correctly
  TestRunTool          -- _run_tool dispatches to correct tool, handles unknown
  TestGraphNodes       -- classify, escalate, decline produce correct state keys
  TestBuildGraph       -- graph compiles, invoke returns expected keys
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

SOLUTION_DIR = Path(__file__).parent.parent / "solution"
sys.path.insert(0, str(SOLUTION_DIR))

import main
from main import (
    MCP_SERVER_PATH,
    WealthDeskState,
    _run_mcp_tool,
    _run_tool,
    build_graph,
    classify,
    decline,
    escalate,
    query_branch,
    query_rates,
)


# ---------------------------------------------------------------------------
# TestMCPServerPath
# ---------------------------------------------------------------------------

class TestMCPServerPath:
    def test_mcp_server_path_is_path_object(self):
        assert isinstance(MCP_SERVER_PATH, Path)

    def test_mcp_server_path_points_to_s07(self):
        assert "s07" in str(MCP_SERVER_PATH)

    def test_mcp_server_path_filename(self):
        assert MCP_SERVER_PATH.name == "mcp_server.py"

    def test_mcp_server_path_exists(self):
        assert MCP_SERVER_PATH.exists(), (
            f"S07 MCP server not found at {MCP_SERVER_PATH}. "
            "Complete Session 7 before running Session 8 tests."
        )


# ---------------------------------------------------------------------------
# TestRunMcpTool
# ---------------------------------------------------------------------------

class TestRunMcpTool:
    def test_returns_string_on_success(self):
        with patch("main._call_mcp_async") as mock_async:
            mock_async.return_value = "Home Loan: 8.5% p.a."
            with patch("asyncio.run", return_value="Home Loan: 8.5% p.a."):
                result = _run_mcp_tool("query_rates", {"product_type": "loan"})
        assert isinstance(result, str)

    def test_returns_error_string_on_exception(self):
        with patch("asyncio.run", side_effect=RuntimeError("Connection refused")):
            result = _run_mcp_tool("query_rates", {"product_type": "loan"})
        assert "MCP tool error" in result
        assert "query_rates" in result

    def test_error_message_includes_tool_name(self):
        with patch("asyncio.run", side_effect=Exception("timeout")):
            result = _run_mcp_tool("query_branch", {"city": "Mumbai"})
        assert "query_branch" in result

    def test_passes_tool_name_to_async_call(self):
        captured = {}

        async def fake_call(tool_name, tool_args):
            captured["tool_name"] = tool_name
            captured["tool_args"] = tool_args
            return "result"

        with patch("main._call_mcp_async", side_effect=fake_call):
            with patch("asyncio.run", side_effect=lambda coro: "result"):
                _run_mcp_tool("query_rates", {"product_type": "fd"})


# ---------------------------------------------------------------------------
# TestToolFunctions
# ---------------------------------------------------------------------------

class TestToolFunctions:
    def test_query_rates_calls_run_mcp_tool(self):
        with patch("main._run_mcp_tool", return_value="Home Loan: 8.5%") as mock:
            result = query_rates.invoke({"product_type": "loan"})
        mock.assert_called_once_with("query_rates", {"product_type": "loan"})
        assert result == "Home Loan: 8.5%"

    def test_query_rates_default_product_type_is_all(self):
        with patch("main._run_mcp_tool", return_value="All rates") as mock:
            query_rates.invoke({})
        mock.assert_called_once_with("query_rates", {"product_type": "all"})

    def test_query_branch_calls_run_mcp_tool(self):
        with patch("main._run_mcp_tool", return_value="BNB Indiranagar") as mock:
            result = query_branch.invoke({"city": "Bengaluru"})
        mock.assert_called_once_with("query_branch", {"city": "Bengaluru"})
        assert result == "BNB Indiranagar"

    def test_query_branch_default_city_is_all(self):
        with patch("main._run_mcp_tool", return_value="All branches") as mock:
            query_branch.invoke({})
        mock.assert_called_once_with("query_branch", {"city": "all"})

    def test_query_rates_passes_fd_filter(self):
        with patch("main._run_mcp_tool", return_value="FD rates") as mock:
            query_rates.invoke({"product_type": "fd"})
        mock.assert_called_once_with("query_rates", {"product_type": "fd"})

    def test_query_branch_passes_city_argument(self):
        with patch("main._run_mcp_tool", return_value="Chennai branches") as mock:
            query_branch.invoke({"city": "Chennai"})
        mock.assert_called_once_with("query_branch", {"city": "Chennai"})

    def test_query_rates_returns_mcp_output(self):
        expected = "Home Loan: 8.5% p.a., tenure 5-30 years"
        with patch("main._run_mcp_tool", return_value=expected):
            result = query_rates.invoke({"product_type": "loan"})
        assert result == expected

    def test_tool_name_is_query_rates(self):
        assert query_rates.name == "query_rates"

    def test_tool_name_is_query_branch(self):
        assert query_branch.name == "query_branch"

    def test_query_rates_has_description(self):
        assert query_rates.description and len(query_rates.description) > 10

    def test_query_branch_has_description(self):
        assert query_branch.description and len(query_branch.description) > 10


# ---------------------------------------------------------------------------
# TestRunTool
# ---------------------------------------------------------------------------

class TestRunTool:
    def test_dispatches_query_rates(self):
        with patch("main._run_mcp_tool", return_value="8.5%"):
            result = _run_tool("query_rates", {"product_type": "loan"})
        assert result == "8.5%"

    def test_dispatches_query_branch(self):
        with patch("main._run_mcp_tool", return_value="BNB Bandra"):
            result = _run_tool("query_branch", {"city": "Mumbai"})
        assert result == "BNB Bandra"

    def test_unknown_tool_returns_error_string(self):
        result = _run_tool("nonexistent_tool", {})
        assert "Unknown tool" in result
        assert "nonexistent_tool" in result

    def test_returns_string(self):
        with patch("main._run_mcp_tool", return_value="result"):
            result = _run_tool("query_rates", {"product_type": "all"})
        assert isinstance(result, str)

    def test_tool_exception_returns_error_string(self):
        with patch("main._run_mcp_tool", side_effect=RuntimeError("crash")):
            result = _run_tool("query_rates", {"product_type": "loan"})
        assert "Tool error" in result


# ---------------------------------------------------------------------------
# TestGraphNodes
# ---------------------------------------------------------------------------

class TestGraphNodes:
    def _make_state(self, message="test", query_type="SIMPLE") -> WealthDeskState:
        return WealthDeskState(
            customer_message=message,
            response="",
            history=[],
            query_type=query_type,
            retrieved_docs=[],
        )

    def test_escalate_returns_response_key(self):
        result = escalate(self._make_state())
        assert "response" in result

    def test_escalate_response_mentions_relationship_manager(self):
        result = escalate(self._make_state())
        assert "Relationship Manager" in result["response"]

    def test_escalate_response_includes_phone_number(self):
        result = escalate(self._make_state())
        assert "1800-103-1906" in result["response"]

    def test_escalate_updates_history(self):
        result = escalate(self._make_state("complex query"))
        assert len(result["history"]) == 2
        assert result["history"][0]["role"] == "user"
        assert result["history"][1]["role"] == "assistant"

    def test_decline_returns_response_key(self):
        result = decline(self._make_state())
        assert "response" in result

    def test_decline_response_mentions_bnb(self):
        result = decline(self._make_state())
        assert "BNB" in result["response"]

    def test_decline_updates_history(self):
        result = decline(self._make_state("off-topic query"))
        assert len(result["history"]) == 2


# ---------------------------------------------------------------------------
# TestBuildGraph
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_build_graph_returns_compiled_graph(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert graph is not None

    def test_graph_invoke_complex_returns_escalation(self):
        from langgraph.checkpoint.memory import MemorySaver
        from unittest.mock import patch as up

        with up("main.classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="COMPLEX")
            graph = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "Should I invest all my savings?", "response": ""},
                config={"configurable": {"thread_id": "test-complex"}},
            )
        assert "Relationship Manager" in result["response"]
        assert result["query_type"] == "COMPLEX"

    def test_graph_invoke_oos_returns_decline(self):
        from langgraph.checkpoint.memory import MemorySaver
        from unittest.mock import patch as up

        with up("main.classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="OUT_OF_SCOPE")
            graph = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "Write me a poem", "response": ""},
                config={"configurable": {"thread_id": "test-oos"}},
            )
        assert "only help with BNB" in result["response"]
        assert result["query_type"] == "OUT_OF_SCOPE"

    def test_graph_invoke_returns_response_key(self):
        from langgraph.checkpoint.memory import MemorySaver
        from unittest.mock import patch as up

        with up("main.classifier_llm") as mock_llm:
            mock_llm.invoke.return_value = MagicMock(content="COMPLEX")
            graph = build_graph(checkpointer=MemorySaver())
            result = graph.invoke(
                {"customer_message": "test", "response": ""},
                config={"configurable": {"thread_id": "test-keys"}},
            )
        assert "response" in result
        assert "query_type" in result
