import os

API_KEY = os.environ.get("OPENAI_API_KEY")
API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Export it before running API-based evaluation.")
