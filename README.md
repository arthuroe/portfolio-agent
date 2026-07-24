# Portfolio Agent — RAG Chat Widget

A small RAG (retrieval-augmented generation) chatbot that answers questions about
your background, grounded in your actual CV/portfolio content. Built with FastAPI,
Postgres + pgvector, Voyage AI embeddings, and Claude.

## How it works

1. `content.json` holds your CV/portfolio broken into short chunks.
2. `scripts/embed_content.py` embeds each chunk and stores it in Postgres (pgvector).
3. `app.py` exposes a `POST /ask` endpoint: it embeds the incoming question, finds
   the most relevant chunks, and asks Claude to answer using only that context.
4. `static/chat-widget.html` is a drop-in widget for your GitHub Pages site.

## Setup

### 1. Get a Postgres database with pgvector

Any of these work and have free tiers:

- [Neon](https://neon.tech)
- [Supabase](https://supabase.com)
- [Render Postgres](https://render.com)

Copy the connection string — you'll need it as `DATABASE_URL`.

### 2. Get API keys

- Anthropic: https://console.anthropic.com/
- Voyage AI (embeddings): https://www.voyageai.com/ — free tier is enough for this project

### 3. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# then fill in DATABASE_URL, ANTHROPIC_API_KEY, VOYAGE_API_KEY
```

### 5. Edit your content

Update `content.json` with your own CV/project details — the file included here
is seeded with Arthur's CV as a starting point. Keep each chunk focused (one job,
one project, one skills group) so retrieval stays precise.

### 6. Load content into the database

```bash
python scripts/embed_content.py
```

Re-run this any time you update `content.json`.

### 7. Run locally

```bash
uvicorn app:app --reload --port 8000
```

Test it:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What backend work has Arthur done?"}'
```

### 8. Deploy

Render and Fly.io both have free/cheap tiers that work well for something this
light. Either way:

- Set the same environment variables from `.env` in your host's dashboard
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

# Portfolio Agent — RAG Chat Widget

A small RAG (retrieval-augmented generation) chatbot that answers questions about
your background, grounded in your actual CV/portfolio content. Built with FastAPI,
Postgres + pgvector, Voyage AI embeddings, and Groq (Llama 3.3 70B) for generation.

Every piece here runs on a free tier — Postgres (Neon), embeddings (Voyage),
and generation (Groq) — so this project costs $0 to build and run at this scale.

## How it works

1. `content.json` holds your CV/portfolio broken into short chunks.
2. `scripts/embed_content.py` embeds each chunk and stores it in Postgres (pgvector).
3. `app.py` exposes a `POST /ask` endpoint: it embeds the incoming question, finds
   the most relevant chunks, and asks a Groq-hosted model to answer using only
   that context.
4. `static/chat-widget.html` is a drop-in widget for your GitHub Pages site.

## Setup

### 1. Get a Postgres database with pgvector

Any of these work and have free tiers:

- [Neon](https://neon.tech)
- [Supabase](https://supabase.com)
- [Render Postgres](https://render.com)

Copy the connection string — you'll need it as `DATABASE_URL`.

### 2. Get API keys

- Groq (generation): https://console.groq.com/ — free tier, no credit card required
- Voyage AI (embeddings): https://www.voyageai.com/ — free tier is enough for this project

### 3. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# then fill in DATABASE_URL, GROQ_API_KEY, VOYAGE_API_KEY
```

### 5. Edit your content

Update `content.json` with your own CV/project details — the file included here
is seeded with Arthur's CV as a starting point. Keep each chunk focused (one job,
one project, one skills group) so retrieval stays precise.

### 6. Load content into the database

```bash
python scripts/embed_content.py
```

Re-run this any time you update `content.json`.

### 7. Run locally

```bash
uvicorn app:app --reload --port 8000
```

Test it:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What backend work has Arthur done?"}'
```

### 8. Deploy

Render and Fly.io both have free/cheap tiers that work well for something this
light. Either way:

- Set the same environment variables from `.env` in your host's dashboard
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

### 9. Add the widget to your portfolio site

Copy the contents of `static/chat-widget.html` into your GitHub Pages HTML
(just before `</body>`), and update `API_URL` at the top of the script to point
to your deployed API.

## Notes

- The system prompt restricts answers to the provided context, so the bot
  shouldn't invent skills or experience you don't have.
- `TOP_K` in `app.py` controls how many chunks are retrieved per question —
  4 is a reasonable default for a CV-sized content set.
- CORS is locked down via `ALLOWED_ORIGINS` — set this to your actual GitHub
  Pages domain before deploying, not `*`.

### 9. Add the widget to your portfolio site

Copy the contents of `static/chat-widget.html` into your GitHub Pages HTML
(just before `</body>`), and update `API_URL` at the top of the script to point
to your deployed API.

## Notes

- The system prompt restricts answers to the provided context, so the bot
  shouldn't invent skills or experience you don't have.
- `TOP_K` in `app.py` controls how many chunks are retrieved per question —
  4 is a reasonable default for a CV-sized content set.
- CORS is locked down via `ALLOWED_ORIGINS` — set this to your actual profile domain before deploying, not `*`.
