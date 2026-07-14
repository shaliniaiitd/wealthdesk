"""LangGraph entrypoint for the US-03 documents agent."""
import sqlite3
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from config import DATA_DIR, CHECKPOINT_DB
from nodes import respond
from state import WealthDeskState


def build_graph(checkpointer=None):
    builder = StateGraph(WealthDeskState)
    builder.add_node("respond", respond)
    builder.set_entry_point("respond")
    builder.add_edge("respond", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


graph = build_graph()


def run() -> None:
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
    
    graph_with_memory = build_graph(checkpointer=SqliteSaver(conn))
   
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("=" * 55)
    print("  WealthDesk Documents Agent | Bharat National Bank")
    print("  Type 'quit' to exit")
    print("=" * 55)
    print(f"  Session: {thread_id[:8]}...")
    print("=" * 55)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nWealthDesk: Session ended. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("\nWealthDesk: Thank you for choosing Bharat National Bank. Goodbye!")
            break

    
        result = graph_with_memory.invoke(
            {"customer_message": user_input, "response": ""}, config = config)
       
        print(f"\nWealthDesk: {result['response']}")


if __name__ == "__main__":
    run()
