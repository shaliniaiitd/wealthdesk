"""
wealthdesk/nodes.py


Graph nodes for the US-03 documents agent.

"""
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import SYSTEM_PROMPT
from state import WealthDeskState
from tools import llm, retrieve_context


def respond(state: WealthDeskState) -> dict:
    """Build a message list, add retrieved context, and generate a response."""
    history = state.get("history", [])
    #Step A: Build the message list for the LLM call.
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    #   Then loop over `history` and append each turn:
    #     - {"role": "user", ...}      → HumanMessage(content=turn["content"])
    #     - {"role": "assistant", ...}

    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))

    # RAG CHANGES
    #   # Step B: Retrieve relevant context based on the customer's message from D 
    # 
    retrieved_context = retrieve_context(state["customer_message"])
    if retrieved_context:
        messages.append(HumanMessage(content=("Retrieved policy context:\n"
                                              f"{retrieved_context}\n\n"
                                              f"Customer question: {state['customer_message']}") ))
    else:
        messages.append(HumanMessage(content=state["customer_message"]))
    # RAG CHANGES

    try:
        result = llm.invoke(messages)
        response_text = result.content
    except Exception as exc:
        print(f"[WealthDesk] LLM error: {exc}")
        response_text = "I am temporarily unavailable. Please try again in a moment."

    new_history = history + [
        {"role": "user", "content": state["customer_message"]},
        {"role": "assistant", "content": response_text},
    ]
    return {"response": response_text, "history": new_history}
