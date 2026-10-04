import os
import requests
from dotenv import load_dotenv

load_dotenv()

NUGEN_API_KEY = os.getenv("NUGEN_API_KEY")
NUGEN_API_URL = os.getenv("NUGEN_API_URL")
NUGEN_MODEL = os.getenv("NUGEN_MODEL")


def build_prompt(question, documents):
    context_parts = []

    for i, item in enumerate(documents, start=1):
        document = item["document"]
        text = document.payload.get("text", "")
        page = document.payload.get("page", "")

        context_parts.append(
            f"[Context {i} | Page {page}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    return f"""
Answer the question using only the provided context.
If the answer is not present in the context, say that it is not available.

Context:
{context}

Question:
{question}

Answer:
"""


def generate_answer(question, documents):

    if not NUGEN_API_KEY:
        raise ValueError("NUGEN_API_KEY is missing in .env")

    if not NUGEN_API_URL:
        raise ValueError("NUGEN_API_URL is missing in .env")

    if not NUGEN_MODEL:
        raise ValueError("NUGEN_MODEL is missing in .env")

    if not NUGEN_API_URL.startswith("http://") and not NUGEN_API_URL.startswith("https://"):
        raise ValueError(f"Invalid NUGEN_API_URL: {NUGEN_API_URL}")

    prompt = build_prompt(question, documents)

    headers = {
        "Authorization": f"Bearer {NUGEN_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": NUGEN_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 300,
        "temperature": 0.3,
        "stream": False
    }

    response = requests.post(
        NUGEN_API_URL,
        headers=headers,
        json=payload,
        timeout=60
    )

    if not response.ok:
        raise RuntimeError(
            f"Nugen API error {response.status_code}: {response.text}"
        )

    data = response.json()

    return data["choices"][0]["message"]["content"]