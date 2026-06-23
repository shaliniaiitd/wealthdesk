"""
s02/tests/conftest.py
---------------------
Pytest configuration for Session 2 tests.

Sets dummy environment variables before any test module is imported.
Without GROQ_API_KEY, the module-level guard in solution/main.py raises
ValueError during test collection and aborts pytest before a single test runs.

The key values are never sent to any external service because every test
that calls the LLM mocks main.llm.
"""
import os
import sys

sys.modules.pop("main", None)

os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
os.environ.setdefault("LANGSMITH_API_KEY", "test-langsmith-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
