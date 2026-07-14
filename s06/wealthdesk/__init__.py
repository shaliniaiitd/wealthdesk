MOVED TO CONFIG.PY so as to allow run as agent.py and not  as a module run python -m <>wealhdesk.agent
"""
WealthDesk package -- US 04:
==========================================================

This file runs automatically when Python imports the wealthdesk package.
Use it to set up the environment before any other module loads.
"""
# import os

# os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# # ---------------------------------------------------------------------------
# #  Environment setup
# # ---------------------------------------------------------------------------
# # Same as Session 1: import and call load_dotenv() so GROQ_API_KEY is
# # available before tools.py tries to read it.
# #
# from dotenv import load_dotenv
# load_dotenv()
