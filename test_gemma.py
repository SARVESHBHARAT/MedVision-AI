import os
from google import genai

from dotenv import load_dotenv
load_dotenv()

try:
    print(f"genai version: {getattr(genai, '__version__', 'unknown')}")
except Exception:
    print("genai version: unknown")

api_key = os.getenv("GEMMA_API_KEY") or os.getenv("GEMINI_API_KEY")
print(f"API Key present: {bool(api_key)}")

client = genai.Client(api_key=api_key)
print("Client initialized.")

try:
    print("Testing 'gemma-4-26b-a4b-it'...")
    res = client.models.generate_content(
        model='gemma-4-26b-a4b-it',
        contents='Reply with exactly: MEDVISION GEMMA TEST OK'
    )
    print("Response:")
    print(res.text)
except Exception as e:
    print(f"Exception for 'gemma-4-26b-a4b-it': {type(e).__name__}: {e}")

try:
    print("\nTesting 'gemini-3.8-flash' as fallback to verify project access...")
    res = client.models.generate_content(
        model='gemini-3.8-flash',
        contents='Reply with exactly: MEDVISION GEMMA TEST OK'
    )
    print("Response:")
    print(res.text)
except Exception as e:
    print(f"Exception for 'gemini-3.8-flash': {type(e).__name__}: {e}")

