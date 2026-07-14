"""Configuration for the US-04
"""
from pathlib import Path
from dotenv import load_dotenv
import os

os.environ.setdefault("HF_HUB_VERBOSITY", "error")
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"  #GPT OSS 120B"  #"Qwen3.6 27B"
TEMPERATURE = 0.3
MAX_TOKENS = 300

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
