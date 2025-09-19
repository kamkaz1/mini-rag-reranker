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

## Example Requests

### Easy Example (Baseline Mode - No Answer)
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"q": "What is ISO 13849-1?", "k": 5, "mode": "baseline"}'
```

**Result**: Returns contexts but no answer
```json
{
    "answer": null,
    "contexts": [
        {
            "text": "ISO 13849-1:2015 - Safety of machinery - Safety-related parts of control systems...",
            "score": 0.364,
            "source": "ISO 13849-1:2015 - Safety of machinery - Safety-related parts of control systems",
            "url": "https://www.iso.org/standard/69883.html"
        }
    ],
    "reranker_used": false,
    "reason": "Top result score (0.364) below confidence threshold (0.7)",
    "query": "What is ISO 13849-1?"
}
```

### Tricky Example (Reranked Mode - Complete Answer)
```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"q": "What are the different performance levels in safety systems?", "k": 5, "mode": "reranked"}'
```

**Result**: Returns complete answer with citations
```json
{
    "answer": "Performance levels are determined based on the combination of safety integrity level (SIL) and mean time to dangerous failure (MTTFd) The standard defines performance levels (PL) from PLa to PLe, where PLe represents the highest level of safety integrity...\n\nSources: [1] ISO 13849-1:2015 - Safety of machinery - Safety-related parts of control systems, [2] OSHA Machine Guarding Safety Requirements",
    "contexts": [
        {
            "text": "ISO 13849-1:2015 - Safety of machinery - Safety-related parts of control systems...",
            "score": 1.0,
            "source": "ISO 13849-1:2015 - Safety of machinery - Safety-related parts of control systems",
            "url": "https://www.iso.org/standard/69883.html"
        }
    ],
    "reranker_used": true,
    "reason": "Answer generated from 2 sources with top score 1.000",
    "query": "What are the different performance levels in safety systems?"
}
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
