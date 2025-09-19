# Mini RAG + Reranker System

A small question-answering service over industrial safety documents with baseline similarity search and hybrid reranking.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize system (processes documents and creates embeddings)
python setup.py

# Start the API server
uvicorn main:app --reload
```

## Usage

```bash
# Test with curl
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"q": "What is ISO 13849-1?", "k": 10, "mode": "baseline"}'

# Run comprehensive test suite
python test_comparison.py
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

## Results

| Mode | Questions Answered | Success Rate | Avg Response Time |
|------|-------------------|--------------|-------------------|
| Baseline | 5/8 | 62.5% | 0.118s |
| Reranked | 8/8 | 100.0% | 0.009s |

**Improvement**: +37.5% answer coverage with hybrid reranking

## Key Learnings

**Hybrid Reranking Effectiveness**: The combination of semantic (vector) and lexical (BM25) signals significantly improved answer quality and coverage. The hybrid approach with α=0.6 (60% vector, 40% BM25) consistently boosted confidence scores and enabled answers when baseline vector search failed. This demonstrates that pure semantic similarity can miss important keyword matches, while pure keyword matching lacks semantic understanding.

**Score Normalization and Confidence Thresholds**: Proper score normalization between different ranking signals was crucial for meaningful blending. Setting an appropriate confidence threshold (0.7) prevented low-quality answers while the reranker's score boosting allowed more questions to meet this threshold. The system's ability to abstain when uncertain, rather than providing poor answers, proved valuable for maintaining answer quality.
