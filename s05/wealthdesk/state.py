"""
wealthdesk/state.py
State for the US-03 documents agent."""

from typing import TypedDict

class WealthDeskState(TypedDict):
    customer_message: str
    response: str
    history: list[dict]
   