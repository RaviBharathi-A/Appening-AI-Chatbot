# Agentic AI eBook RAG Chatbot

A Python chatbot that retrieves evidence from the Agentic AI eBook and returns a source-grounded response through FastAPI. It uses **LangGraph**, a persistent **local SQLite TF-IDF vector index**, and extractive answer generation. It runs without OpenAI or Pinecone credentials, network access to model APIs, or hosted vector services.

## Assessment requirement mapping

| Assessment area | Implementation |
| --- | --- |
| PDF ingestion and chunking | `PyPDFLoader` plus `RecursiveCharacterTextSplitter` (1,000-character chunks, 200-character overlap) |
| Embedding/vector retrieval | Local sparse TF-IDF vectors persisted in SQLite; cosine similarity ranks candidate chunks |
| Graph orchestration | LangGraph retrieval, relevance gate, extractive answer, and grounding check |
| API response | FastAPI `POST /chat` returns `query`, `final_answer`, `retrieved_context_chunks`, and `confidence_score` |
| Sample validation | Six assessment-style questions, including an out-of-scope refusal, in `tests_sample_queries.py` |

## Provider and vector-store decision

The supplied assessment example uses OpenAI dense embeddings and Pinecone. During setup, the OpenAI embeddings endpoint returned HTTP 401 (`invalid_api_key`) for the configured credential, and Pinecone setup was also a blocker. Rather than claim an unverified external integration works, this submission keeps its core RAG path runnable without either service: retrieval uses local TF-IDF term vectors and the answer is composed only from exact sentences in the retrieved eBook chunks.

This is a deliberate, transparent alternative—not an OpenAI-generated response and not a Pinecone index. It avoids embedding API costs and credentials while providing reproducible retrieval, source citations, a measurable retrieval score, and a deterministic refusal when evidence is weak. The trade-off is lexical rather than semantic retrieval: paraphrased questions may retrieve less well than dense embeddings. The existing API key is not read, displayed, or required by this local implementation. Do not publish API keys.

## Requirements

- Python 3.10 or newer
- The supplied PDF, by default `D:\Placement\Ebook-Agentic-AI.pdf`
- No API keys, account setup, or hosted vector database

## Install (Windows PowerShell)

Open PowerShell in the repository directory:

```powershell
cd "C:\Appening AI Chatbot\Appening-AI-Chatbot"
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional local settings are in `.env.example`. Copy the template if you want to change the database path, collection, result count, or relevance threshold:

```powershell
Copy-Item .env.example .env
notepad .env
```

No secret values belong in `.env` for the current local-only implementation.

## Ingest the eBook

```powershell
.\.venv\Scripts\python.exe -m src.ingestion --pdf-path "D:\Placement\Ebook-Agentic-AI.pdf"
```

For another PDF, replace the path:

```powershell
.\.venv\Scripts\python.exe -m src.ingestion --pdf-path "C:\path\to\Ebook-Agentic-AI.pdf"
```

Ingestion extracts page text, splits it into overlapping chunks, and stores the chunks and page/source metadata in `data/agentic-ai-vectors.sqlite3`. It builds sparse term vectors locally; it does not call an embedding API. The first run creates the database. Subsequent runs atomically replace the selected local collection. The database is ignored by Git.

## Run the chatbot API

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs`. `GET /health` reports whether local document chunks are ready. Try `POST /chat` with:

```json
{
  "query": "What is Agentic AI according to the eBook?"
}
```

Example response shape:

```json
{
  "query": "What is Agentic AI according to the eBook?",
  "final_answer": "An exact source sentence selected from the retrieved passage.",
  "retrieved_context_chunks": [
    "The source passage text. (Source: Ebook-Agentic-AI.pdf, page 2)"
  ],
  "confidence_score": 0.51
}
```

The answer is extractive: it consists only of sentences copied from the retrieved source chunks. If no chunk meets the relevance threshold, or the graph's exact-source grounding check fails, the chatbot returns `I cannot answer based on the provided document.` with a zero confidence score. `confidence_score` is the mean cosine relevance of retained chunks, not a guarantee of factual correctness.

The provided PDF has some custom-font extraction warnings and flattened table layouts. The app preserves source wording and page metadata rather than silently repairing or inventing text; inspect the returned cited chunks when validating answers that depend on table content.

## Run tests and benchmark questions

Run local unit tests without API calls:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

With the API running in another terminal, run the six sample questions:

```powershell
.\.venv\Scripts\python.exe tests_sample_queries.py
```

The sample script checks response fields, bounded confidence values, answers to the in-scope prompts, and refusal for the out-of-scope France question.

## Architecture

```text
PDF → PyPDFLoader → RecursiveCharacterTextSplitter
    → local TF-IDF term vectors → SQLite collection

POST /chat → LangGraph retrieve → relevance gate → extract source sentences
                                                → exact-source grounding check
                                                → response or refusal
```

- `src/config.py`: local database and retrieval settings.
- `src/ingestion.py`: PDF parsing, page-aware chunking, and local persistence.
- `src/vector_store.py`: SQLite storage and TF-IDF cosine retrieval.
- `src/graph.py`: LangGraph state, retrieval gate, extractive response, and grounding validation.
- `app.py`: FastAPI endpoints and response schema.
- `tests_sample_queries.py`: assignment-style API validation questions.

## Default local configuration

The optional `.env` settings default to:

- `LOCAL_VECTOR_STORE_PATH=data/agentic-ai-vectors.sqlite3`
- `LOCAL_VECTOR_STORE_COLLECTION=agentic-ai-ebook`
- `RETRIEVAL_TOP_K=8`
- `RETRIEVAL_SCORE_THRESHOLD=0.08`

No OpenAI or Pinecone variables are read by this version.

## Publishing checklist

- Check `git status` and confirm `.env`, PDF files, and `data/*.sqlite3` are not staged.
- Do not commit keys, even if a key is no longer in use; revoke any key that was shared or exposed.
- State clearly in the submission that this runnable version uses local TF-IDF retrieval and extractive answers instead of the assessment's hosted Pinecone/OpenAI embedding path.
