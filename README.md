## NormAI Take Home Challenge

This repository contains a FastAPI service that loads the `docs/laws.pdf` document into an in-memory vector store (Qdrant via LlamaIndex) and answers natural-language questions using OpenAI.

The service exposes one endpoint:

- `POST /query` — accepts a JSON body with a `query` string and returns an answer with citations from the document.

The included Docker setup runs only the FastAPI backend (no frontend is required).

---

### Prerequisites

- Docker and Docker Compose
- An OpenAI API key with access to the specified models

Optional for local (non-Docker) runs:

- Python 3.11+

---

### Quickstart (Docker Compose)

1. Create a `.env` file in the repository root:

```
OPENAI_API_KEY=your_openai_key_here
```

2. Build and start the API:

```
docker compose up --build
```

3. Verify the API is running at `http://localhost:8000`.

- Interactive docs (Swagger): `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

4. Query the service:

```
curl -sS -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"what happens if I steal from the Sept?"}'
```

The response includes the original `query`, an LLM-generated `response`, and `citations` from the PDF, for example:

```
{
  "query": "what happens if I steal from the Sept?",
  "response": "…generated answer based on the document…",
  "citations": [
    { "source": "Law 4.1.2", "text": "…relevant excerpt…" },
    { "source": "Law 4.1.3", "text": "…relevant excerpt…" }
  ]
}
```

To stop:

```
docker compose down
```

Notes:

- The vector DB (Qdrant) is in-memory; data is rebuilt on startup/request.
- The compose file mounts `./app` and `./docs` read-only into the container.

---

### Local Development (without Docker)

1. Create and activate a virtual environment (recommended):

```
python3.11 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```
pip install --upgrade pip
pip install -r requirements.txt
```

3. Set your OpenAI API key in the environment:

```
export OPENAI_API_KEY=your_openai_key_here
```

4. Start the server:

```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

5. Test the endpoint (same as above):

```
curl -sS -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"what happens if I steal from the Sept?"}'
```

---

### API Reference

- `POST /query`
  - Request body:
    - `query` (string): The user question.
  - Response body:
    - `query` (string): Echo of the question.
    - `response` (string): Model-generated answer using context from the PDF.
    - `citations` (array): Up to top-k supporting excerpts with `source` and `text`.

Example request body:

```
{
  "query": "what happens if I steal from the Sept?"
}
```

---

### Troubleshooting

- Missing or invalid `OPENAI_API_KEY` will result in initialization or runtime errors. Ensure it is set in your shell or `.env` before starting.
- If port `8000` is in use, change the published port in `docker-compose.yml` or in the `uvicorn` command.
- PDF parsing relies on `docs/laws.pdf`. Ensure the file exists and is readable.

---

### Project Structure (relevant parts)

```
app/
  main.py          # FastAPI entrypoint and /query endpoint
  utils.py         # Document parsing, Qdrant/LlamaIndex integration
docs/
  laws.pdf         # Source document used to build the vector index
Dockerfile         # API container image
docker-compose.yml # API-only compose definition
requirements.txt   # Python dependencies
```
