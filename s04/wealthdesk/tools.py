"""LLM and ChromaDB retrieval helpers for the US-03 documents agent."""
import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from config import EMBED_MODEL, ENABLE_RAG, MAX_TOKENS, MODEL_NAME, TEMPERATURE, TOP_K_RESULTS, VECTOR_DIR

load_dotenv()

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
