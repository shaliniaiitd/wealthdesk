"""
WealthDesk -- Session 1: Basic Conversational Agent (US-01)
===========================================================

Your task: fill in every section marked TODO.
The terminal loop and imports are provided.
You are building the LangGraph state, one node, and the graph wiring.

Run when you are done:
    python s01/starter/main.py
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# TODO 1 of 1 in this block:
# Call load_dotenv() here. It reads your .env file and loads every KEY=VALUE
# pair into os.environ so that os.getenv() can find them below.
# Rule: always call load_dotenv() before any os.getenv() calls.

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Check that your .env file exists and contains the key."
    )

MODEL_NAME  = "meta-llama/llama-4-scout-17b-16e-instruct"
TEMPERATURE = 0.3
MAX_TOKENS  = 300

# TODO 2 of 5:
# Write the system prompt for WealthDesk using the four-component structure:
#
#   Persona:          Who WealthDesk is and its tone
#   Domain knowledge: Products and rates (home loan, FD, etc.) + eligibility formulas
#   Rules:            What it must always do, never do, and how to handle edge cases
#   Output format:    Response length limit and sign-off (put this section LAST)
#
# Rates to include:
#   Home Loan 8.5% p.a., Personal Loan 12.0%, FD 1yr 6.8%, FD 2yr 7.1%
#   Eligibility: Home Loan max = monthly income × 60 (e.g. Rs. 80,000 → Rs. 48,00,000)
#                Personal Loan max = monthly income × 24
#
# Hint: use a triple-quoted string ("""...""") -- see 06_strings_participant.md

SYSTEM_PROMPT = """
# TODO: Write the WealthDesk system prompt here.
"""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

# TODO 3 of 5:
# Define WealthDeskState as a TypedDict with two fields:
#   customer_message: str   -- the customer's question
#   response: str           -- the agent's answer
#
# Hint: see 08_typeddict_participant.md for the exact pattern.

class WealthDeskState(TypedDict):
    pass  # TODO: replace 'pass' with the two field definitions


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

# Build the ChatGroq client here (at module level, not inside the function).
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    max_tokens=MAX_TOKENS,
)


def respond(state: WealthDeskState) -> dict:
    """Call the LLM and return the response.

    Node contract:
      - Input: the full WealthDeskState
      - Output: a dict with only the keys this node changed

    TODO 4 of 5:
    1. Build the messages list:
         messages = [
             SystemMessage(content=SYSTEM_PROMPT),
             HumanMessage(content=state["customer_message"]),
         ]
    2. Call llm.invoke(messages) inside a try/except block.
    3. On success: return {"response": result.content}
    4. On exception: print the error and return a safe fallback message.
       Hint: see 10_try_except_participant.md for the exact pattern.
    """
    # TODO: implement this node
    pass


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the WealthDesk graph.

    TODO 5 of 5:
    1. Create a StateGraph: builder = StateGraph(WealthDeskState)
    2. Add the 'respond' node: builder.add_node("respond", respond)
    3. Set the entry point: builder.set_entry_point("respond")
    4. Add an edge from 'respond' to END: builder.add_edge("respond", END)
    5. Return builder.compile()

    Session 1 graph:  START --> respond --> END
    """
    # TODO: build and return the compiled graph
    pass


# ---------------------------------------------------------------------------
# Terminal loop (provided -- no changes needed)
# ---------------------------------------------------------------------------

def run() -> None:
    graph = build_graph()

    print("=" * 55)
    print("  WealthDesk | Bharat National Bank")
    print("  Type 'quit' to exit")
    print("=" * 55)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nWealthDesk: Session ended. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("\nWealthDesk: Thank you for choosing Bharat National Bank. Goodbye!")
            break

        state: WealthDeskState = {
            "customer_message": user_input,
            "response": "",
        }

        result = graph.invoke(state)
        print(f"\nWealthDesk: {result['response']}")


if __name__ == "__main__":
    run()
