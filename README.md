# Mini RAG + Reranker System

A small question-answering service over industrial safety documents with baseline similarity search and hybrid reranking.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Start the API server
uvicorn main:app --reload

# Test with curl
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"q": "What is ISO 13849-1?", "k": 10, "mode": "baseline"}'
```

## API Endpoint

POST `/ask`
- `q`: Question string
- `k`: Number of results (default: 10)
- `mode`: "baseline" or "reranked"

Returns:
- `answer`: Extracted answer with citations or null if uncertain
- `contexts`: Array of relevant chunks with scores and sources
- `reranker_used`: Boolean indicating if reranker was applied
