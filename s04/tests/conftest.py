"""
s04/tests/conftest.py
---------------------
Pytest configuration for Session 4 tests.

Sets dummy API keys and clears any stale 'main' module from a previous session.
HuggingFaceEmbeddings and Chroma are never initialised during tests because
build_graph() calls _init_vectorstore(), which is a no-op when main.vectorstore
has already been patched to a mock object. Tests that call build_graph() always
wrap it with patch("main.vectorstore", mock_vs) so the real vectorstore code
path is never reached.
"""
import os
import sys

sys.modules.pop("main", None)

os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
os.environ.setdefault("LANGSMITH_API_KEY", "test-langsmith-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
