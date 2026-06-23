"""
s03/tests/conftest.py
---------------------
Pytest configuration for Session 3 tests.
Sets dummy environment variables before any test module is imported.
"""
import os
import sys

sys.modules.pop("main", None)

os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
os.environ.setdefault("LANGSMITH_API_KEY", "test-langsmith-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
