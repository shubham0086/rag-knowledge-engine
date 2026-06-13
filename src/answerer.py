"""
Answerer: takes a query + retrieved context chunks, passes them to an LLM,
and returns a grounded answer with citations.
"""
import os
from typing import List, Dict

import anthropic

SYSTEM_PROMPT = """You are a precise knowledge assistant. You answer questions using ONLY the provided context chunks.

Rules:
- Cite your sources using [1], [2] etc. matching the chunk numbers in the context.
- If the answer is not in the context, say "Not found in the provided documents."
- Never hallucinate facts. If you are uncertain, say so.
- Keep answers concise and factual."""


class Answerer:
    def __init__(self, model: str = "claude-opus-4-8-20260528"):
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model  = model

    def answer(self, query: str, context_chunks: List[Dict]) -> Dict:
        if not context_chunks:
            return {
                "answer": "No relevant documents found in the knowledge base.",
                "sources": [],
                "model": self.model,
            }

        context_text = "\n\n".join(
            f"[{i+1}] {c['file']} (score {c['score']}):\n{c['text']}"
            for i, c in enumerate(context_chunks)
        )

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"Context:\n{context_text}\n\nQuestion: {query}"
                }
            ]
        )

        return {
            "answer":  response.content[0].text,
            "sources": [c["file"] for c in context_chunks],
            "model":   self.model,
            "tokens": {
                "input":  response.usage.input_tokens,
                "output": response.usage.output_tokens,
            }
        }
