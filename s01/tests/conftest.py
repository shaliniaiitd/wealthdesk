"""
s01/tests/conftest.py
---------------------
Pytest configuration for Session 1 tests.

Sets a dummy GROQ_API_KEY in the environment before any test module is imported.
Without this, the module-level guard in solution/main.py raises ValueError during
test collection and aborts pytest before a single test runs.

Why a conftest.py rather than removing the guard from main.py?
  The guard in main.py is correct production behaviour: fail fast and loudly if
  the API key is missing. Removing it would teach participants that startup
  validation is optional. Instead we configure the test environment to satisfy
  the guard. The key value is never sent to Groq because every test mocks main.llm.
"""
import os
import sys

# When pytest runs multiple sessions together, 'main' from an earlier session
# stays in sys.modules. Clear it so this session's test file imports from
# its own solution/ directory, not a cached one.
sys.modules.pop("main", None)

os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
os.environ.setdefault("LANGSMITH_API_KEY", "test-langsmith-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
