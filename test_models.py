import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMMA_API_KEY") or os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

print("Fetching available models...")
try:
    for m in client.models.list():
        if "gemma" in m.name.lower() or "gemini" in m.name.lower():
            print(f"Model: {m.name}")
except Exception as e:
    print(f"Failed to list models: {type(e).__name__}: {e}")
