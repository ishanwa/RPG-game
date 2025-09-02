import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

try:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one sentence."}
        ],
        max_tokens=20
    )

    print("✅ API key works! Response:")
    print(response.choices[0].message.content)

except Exception as e:
    print("❌ API key failed:", e)
