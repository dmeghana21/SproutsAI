# test_openai.py (for openai>=1.0.0)

import openai
import os
from dotenv import load_dotenv

# Load key from .env
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Test request using new client-based API
try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10
    )
    print(" API key is working. Response:")
    print(response.choices[0].message.content.strip())

except Exception as e:
    print(" API key failed. Error:")
    print(e)