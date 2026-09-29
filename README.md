# Agentic AI eBook RAG Chatbot

A Python RAG chatbot that answers questions using the Agentic AI eBook, with PDF ingestion, OpenAI embeddings, Pinecone vector search, a LangGraph workflow, and a FastAPI interface. It returns the answer, the source chunks, and an evidence-based confidence score; unrelated or weakly supported questions receive an explicit refusal.

## Requirements

- Python 3.10 or newer
- OpenAI API key
- Pinecone API key and a Pinecone index configured for **1536 dimensions** and **cosine** similarity
- The PDF at `D:\Placement\Ebook-Agentic-AI.pdf`, or another local PDF path

The PDF and API credentials are not committed to this repository.

## Setup (Windows PowerShell)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and provide your OpenAI and Pinecone credentials and the index name. Do not commit `.env`.

Create a Pinecone serverless index in the Pinecone console using dimension `1536`, metric `cosine`, and the cloud/region of your choice. Set the same name in `PINECONE_INDEX_NAME`. The application verifies that the index exists and has the correct dimension; create it in Pinecone before ingestion.

## Ingest the eBook

From the repository root:

```powershell
python -m src.ingestion --pdf-path "D:\Placement\Ebook-Agentic-AI.pdf"
```

To ingest a different copy:

```powershell
python -m src.ingestion --pdf-path "C:\path\to\Ebook-Agentic-AI.pdf"
```

Ingestion extracts page-numbered text, splits it into 1,000-character chunks with 200-character overlap, and upserts the chunks and page/source metadata to Pinecone. Re-running ingestion replaces vectors in the configured index namespace, avoiding duplicate chunks.

## Start the API

```powershell
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation. Health and index readiness are available at `GET /health`; ask a question with `POST /chat`:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/chat `
  -ContentType "application/json" `
  -Body '{"query":"What is Agentic AI?"}'
```

Example response:

```json
{
  "query": "What is Agentic AI?",
  "final_answer": "The answer is based on the retrieved eBook passages.",
  "retrieved_context_chunks": ["Relevant eBook passage (page 3)."],
  "confidence_score": 0.82
}
```

The confidence score is the average normalized relevance of the chunks used for the answer, not a guarantee of factual correctness. If no chunk meets the configured relevance threshold, the chatbot refuses without asking the language model to answer. A graph grader also checks generated answers against the retrieved text and retries once before refusing.

## Try the sample questions

With the API running, in another terminal:

```powershell
python tests_sample_queries.py
```

The script exercises the six assignment topics, including the out-of-scope France question, prints each structured response, and checks the response fields and refusal behavior. It requires the configured services and a previously ingested index.

## Architecture

```text
PDF → PyPDFLoader → RecursiveCharacterTextSplitter
    → OpenAI text-embedding-3-small → Pinecone (page/source metadata)

POST /chat → LangGraph retrieve → relevance gate → generate → grounding grader
                                  ↑                        │
                                  └──── one retry ─────────┘
                                      → grounded response or refusal
```

- `src/config.py`: environment loading, defaults, and index validation.
- `src/ingestion.py`: PDF parsing, page-aware chunking, Pinecone index validation, and vector upsert.
- `src/graph.py`: typed LangGraph state, similarity-threshold retrieval, grounded answer generation, grading, and bounded retry.
- `app.py`: FastAPI request/response schema, chat route, health route, and clear service-error reporting.
- `tests_sample_queries.py`: the six benchmark prompts from the assignment.

The API response follows the requested shape: `query`, `final_answer`, `retrieved_context_chunks`, and `confidence_score`. The chunks include page citations so answers can be checked against the source.

## Configuration

See `.env.example` for all options. `PINECONE_INDEX_NAME`, `OPENAI_API_KEY`, and `PINECONE_API_KEY` are required when using ingestion or chat. `PINECONE_NAMESPACE` defaults to `agentic-ai-ebook`; the same namespace must be used for ingestion and chat. `RETRIEVAL_TOP_K` defaults to 4 and `RETRIEVAL_SCORE_THRESHOLD` to 0.45. Tune the threshold after evaluating retrieval on the supplied eBook; normalized similarity values depend on the embedding/index behavior.

## Security and publishing

Keep `.env`, credentials, and the source PDF out of Git. Before publishing, check that no secrets have been added to the repository. Create and configure the Pinecone index and set API keys locally; GitHub does not receive them.
