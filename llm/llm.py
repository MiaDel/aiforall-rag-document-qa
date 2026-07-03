import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print("API KEY FOUND:", api_key is not None)

client = genai.Client(api_key=api_key)


def generate_answer(question, context):

    prompt = f"""
Answer ONLY from the context.

Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text