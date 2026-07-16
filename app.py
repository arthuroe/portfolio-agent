"""
Minimal RAG API for the portfolio chat widget.

Flow:
  1. User sends a question to POST /ask
  2. We embed the question (Voyage AI)
  3. We find the most similar chunks in Postgres/pgvector
  4. We hand those chunks + the question to Claude, with a system prompt
     that restricts it to answering only from the provided context
  5. We return the answer

Run locally:
    uvicorn app:app --reload --port 8000
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
from dotenv import load_dotenv
import anthropic
import voyageai

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
VOYAGE_API_KEY = os.environ["VOYAGE_API_KEY"]

# Comma-separated list of origins allowed to call this API, e.g.
# "https://yourname.github.io,http://localhost:5500"
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

TOP_K = 4  # how many chunks to retrieve per question

app = FastAPI(title="Portfolio Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST"],
    allow_headers=["*"],
)

voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are a portfolio assistant answering questions about Arthur, \
a backend and data engineer, on behalf of visitors to his portfolio site.

Rules:
- Answer ONLY using the context provided below. Do not invent or assume skills, \
employers, or achievements that are not in the context.
- If the context doesn't contain the answer, say you don't have that information \
and suggest the visitor reach out to Arthur directly.
- Keep answers short (2-4 sentences), friendly, and factual. Speak about Arthur \
in the third person.
- Do not make claims about availability, salary expectations, or personal opinions \
on Arthur's behalf.

Context:
{context}
"""


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


def get_query_embedding(text: str) -> list[float]:
    result = voyage_client.embed([text], model="voyage-3-lite", input_type="query")
    return result.embeddings[0]


def retrieve_chunks(question_embedding: list[float]) -> list[str]:
    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT text FROM content_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (question_embedding, TOP_K),
            )
            rows = cur.fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        q_embedding = get_query_embedding(question)
        chunks = retrieve_chunks(q_embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    context = "\n\n".join(chunks) if chunks else "No relevant context found."

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=SYSTEM_PROMPT.format(context=context),
            messages=[{"role": "user", "content": question}],
        )
        answer = "".join(
            block.text for block in response.content if block.type == "text"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return AskResponse(answer=answer)


@app.get("/health")
def health():
    return {"status": "ok"}
