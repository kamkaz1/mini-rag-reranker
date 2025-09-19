"""
Main FastAPI application for the Mini RAG + Reranker system
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import os

from vector_search import VectorSearch
from reranker import HybridReranker
from answer_generator import AnswerGenerator
from config import DEFAULT_K, RANDOM_SEED
import numpy as np

# Set random seed for reproducibility
np.random.seed(RANDOM_SEED)

app = FastAPI(
    title="Mini RAG + Reranker API",
    description="Question-answering service over industrial safety documents",
    version="1.0.0"
)

# Initialize components
vector_search = VectorSearch()
reranker = HybridReranker()
answer_generator = AnswerGenerator()

# Load vector index on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    print("Starting Mini RAG + Reranker system...")
    
    # Load vector search index
    if not vector_search.load_index():
        raise RuntimeError("Failed to load vector search index. Run setup.py first.")
    
    print("System ready!")


class QueryRequest(BaseModel):
    q: str
    k: int = DEFAULT_K
    mode: str = "baseline"  # "baseline" or "reranked"


class ContextResponse(BaseModel):
    text: str
    score: float
    source: str
    url: str


class QueryResponse(BaseModel):
    answer: Optional[str]
    contexts: List[ContextResponse]
    reranker_used: bool
    reason: str
    query: str


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Ask a question and get an answer with citations
    
    - **q**: The question to ask
    - **k**: Number of results to return (default: 10)
    - **mode**: "baseline" for vector search only, "reranked" for hybrid reranking
    """
    try:
        query = request.q.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        k = max(1, min(request.k, 50))  # Limit k between 1 and 50
        mode = request.mode.lower()
        
        if mode not in ["baseline", "reranked"]:
            raise HTTPException(status_code=400, detail="Mode must be 'baseline' or 'reranked'")
        
        # Perform search based on mode
        if mode == "baseline":
            contexts = vector_search.search(query, k=k)
            reranker_used = False
        else:  # reranked
            contexts = reranker.search_with_reranking(query, vector_search, k=k)
            reranker_used = True
        
        # Generate answer
        response = answer_generator.format_response(query, contexts, reranker_used)
        
        return QueryResponse(**response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vector_index_loaded": vector_search.index is not None,
        "bm25_index_loaded": reranker.bm25 is not None
    }


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        chunk_count = vector_search.index.ntotal if vector_search.index else 0
        return {
            "total_chunks": chunk_count,
            "vector_index_size": chunk_count,
            "bm25_corpus_size": len(reranker.corpus) if reranker.corpus else 0,
            "embedding_model": "all-MiniLM-L6-v2",
            "reranker_type": "hybrid"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
