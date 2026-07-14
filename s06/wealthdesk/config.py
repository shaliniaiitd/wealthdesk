"""Configuration for the US-04
"""
from pathlib import Path
from dotenv import load_dotenv
import os

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

MODEL_NAME = "openai/gpt-oss-20b"  # migrated from deprecated llama-4-scout (Groq deprecation notice)
TEMPERATURE = 0.3
MAX_TOKENS = 600

SYSTEM_PROMPT = """You are WealthDesk, the AI banking assistant at Bharat National Bank (BNB).

Use the retrieved policy context when it is relevant. If the context does not contain the answer,
answer from your general banking knowledge but clearly note that the information is not from the
retrieved policy documents. Never invent product rates or policies that are not present in the
provided context or the system prompt.

Rules:
1. Only discuss BNB products and policies.
2. Keep answers concise and professional.
3. Sign off as: WealthDesk | Bharat National Bank
"""

#################
# DB PATHS
##################
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
# For persisiting MULTI TURN MEMORY
CHECKPOINT_DB = DATA_DIR / "checkpoints.db"
# RAG CHANGES - Data retrival configuration
VECTOR_DIR = DATA_DIR / "vectorstore"
ENABLE_RAG = True
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K_RESULTS = 3
# RAG CHANGES

# SQLLITE TOOLING
DB_PATH = DATA_DIR / "bnb_data.db"

# EVAL (US-05) -----------------------------------------------------------
EVALS_DIR = DATA_DIR / "evals"
GOLDEN_DATASET_PATH = EVALS_DIR / "golden_dataset.json"
LANGSMITH_DATASET_NAME = "wealthdesk-golden-dataset"
EXPERIMENT_PREFIX = "wealthdesk-baseline-eval"
JUDGE_MODEL = "openai/gpt-oss-120b"   # Groq-hosted, free tier, different family from agent model
EVAL_REPETITIONS = 3
PASS_THRESHOLD = 0.75          # 75% mean pass rate (US-05)
VARIANCE_CEILING_PP = 8.0      # flag if stdev across 3 runs exceeds 8 percentage points
