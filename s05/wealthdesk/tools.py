"""LLM and ChromaDB retrieval helpers for the US-03 documents agent."""
import os

#from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBED_MODEL, ENABLE_RAG, MAX_TOKENS, MODEL_NAME, TEMPERATURE, TOP_K_RESULTS, VECTOR_DIR

#load_dotenv()  moved to config instead

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found.\n"
        "Did you copy .env.example to .env and fill in your key?\n"
        "  Windows:  copy .env.example .env\n"
        "  Mac/Linux: cp .env.example .env"
    )

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)

"""PRD Section 1 explicitly draws this distinction:

Unstructured data (ChromaDB): "Retrieved via RAG" → not a tool call, 
always-run retrieval-then-generate.

"""
def build_retriever():
    """Create a Chroma retriever from the persisted vector store."""
    if not ENABLE_RAG or not VECTOR_DIR.exists():
        return None

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = Chroma(
            persist_directory=str(VECTOR_DIR),
            embedding_function=embeddings,
        )
        return vectorstore.as_retriever(search_kwargs={"k": TOP_K_RESULTS})
    except Exception as exc:  # pragma: no cover - safety fallback
        print(f"[WealthDesk] RAG init error: {exc}")
        return None


retriever = build_retriever()


def retrieve_context(question: str) -> str:
    """Return a compact string of matching document chunks for the question."""
    if not ENABLE_RAG or retriever is None:
        return ""

    try:
        docs = retriever.invoke(question)
        if not docs:
            return ""

        snippets = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            text = doc.page_content.strip().replace("\n", " ")
            snippets.append(f"[{source}] {text}")
        return "\n\n".join(snippets[:TOP_K_RESULTS])
    except Exception as exc:  # pragma: no cover - safety fallback
        print(f"[WealthDesk] RAG retrieval error: {exc}")
        return ""
''' As per PRD Section 1 :

Structured data (SQLite): "Queried via tool calls" → query_rates/query_branch as @tool, 
LLM decides when to call.

'''
#SQLITE TOOL
import sqlite3
from langchain_core.tools import tool
from config import DB_PATH

@tool
def query_rates(product_type: str, tenure: int | None = None) -> dict:
    """Get current interest rate for a loan or FD product. product_type is
    e.g. 'home_loan', 'personal_loan', or 'fd'. For 'fd', pass tenure in years."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        if product_type == "fd" and tenure is not None:
            row = conn.execute(
                "SELECT * FROM fd_products WHERE tenure_months = ?", (tenure * 12,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM loan_products WHERE product_id = ?", (product_type,)
            ).fetchone()
        if not row:
            return {"found": False, "message": f"No rate data for '{product_type}'."}
        return {"found": True, **dict(row)}
    finally:
        conn.close()

@tool
def query_branch(city: str) -> dict:
    """Get BNB branch contact info for a given city."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM branches WHERE city = ?", (city,)).fetchall()
        if not rows:
            return {"found": False, "message": f"No branch found in '{city}'."}
        return {"found": True, "branches": [dict(r) for r in rows]}
    finally:
        conn.close()

llm_with_tools = llm.bind_tools([query_rates, query_branch])
