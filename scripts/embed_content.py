"""
Embeds content.json chunks and loads them into a Postgres table using pgvector.
Run this once at setup, and again any time content.json changes.

Requires: pip install psycopg2-binary anthropic python-dotenv
Requires Postgres with the pgvector extension enabled:
    CREATE EXTENSION IF NOT EXISTS vector;
"""

import os
import json
import psycopg2
from dotenv import load_dotenv
import anthropic

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# NOTE: Anthropic does not currently offer a native embeddings endpoint.
# This script uses Voyage AI (Anthropic's recommended embeddings partner).
# Sign up for a free key at https://www.voyageai.com/ and set VOYAGE_API_KEY.
import voyageai

voyage_client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

EMBEDDING_DIM = 1024  # voyage-3-lite output dimension


def get_embedding(text: str) -> list[float]:
    result = voyage_client.embed([text], model="voyage-3-lite", input_type="document")
    return result.embeddings[0]


def setup_table(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS content_chunks (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding vector({EMBEDDING_DIM})
            );
        """)
    conn.commit()


def load_content(conn, chunks: list[dict]):
    with conn.cursor() as cur:
        for chunk in chunks:
            embedding = get_embedding(chunk["text"])
            cur.execute(
                """
                INSERT INTO content_chunks (id, text, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET text = EXCLUDED.text, embedding = EXCLUDED.embedding;
                """,
                (chunk["id"], chunk["text"], embedding),
            )
            print(f"Loaded chunk: {chunk['id']}")
    conn.commit()


def main():
    with open("content.json") as f:
        chunks = json.load(f)

    conn = psycopg2.connect(DATABASE_URL)
    setup_table(conn)
    load_content(conn, chunks)
    conn.close()
    print(f"\nDone. Loaded {len(chunks)} chunks into content_chunks table.")


if __name__ == "__main__":
    main()
