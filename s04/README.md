# US-03 Documents Agent (ChromaDB RAG)

# RAG CHANGES
This module adds the new ChromaDB document retrieval flow for US-03.
# RAG CHANGES

This folder contains a standalone version of the WealthDesk agent that adds ChromaDB-backed document retrieval on top of the S02-style conversational flow.

## What changed
- Added a ChromaDB retriever that reads from the existing data/vectorstore folder.
- Injects the top matching policy chunks into the LLM prompt before generating an answer.
- Keeps a simple conversation history so follow-up questions can be answered in context.

## Prerequisites
1. Install requirements from the repository root:
   ```bash
   pip install -r requirements.txt
   ```
2. Make sure the `.env` file exists and contains a valid GROQ API key.
3. Build the vector store once if it does not exist yet:
   ```bash
   python data/ingest.py
   ```

## Run
From the repository root:
```bash
python -m s03_documents_agent.wealthdesk.agent
```

If you want to run it through LangGraph tooling, use the folder's langgraph.json file.
