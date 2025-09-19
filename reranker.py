"""
Hybrid reranker combining vector similarity with BM25 keyword matching
"""
import sqlite3
import re
from typing import List, Dict
from rank_bm25 import BM25Okapi
from config import DATABASE_PATH, ALPHA, CANDIDATE_K


class HybridReranker:
    def __init__(self):
        self.bm25 = None
        self.corpus = []
        self.chunk_ids = []
        self._build_bm25_index()
    
    def _build_bm25_index(self):
        """Build BM25 index from all chunks in database"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, chunk_text FROM chunks ORDER BY id')
        chunks = cursor.fetchall()
        conn.close()
        
        if not chunks:
            print("No chunks found for BM25 index")
            return
        
        # Tokenize texts for BM25
        tokenized_corpus = []
        for chunk_id, text in chunks:
            tokens = self._tokenize(text)
            tokenized_corpus.append(tokens)
            self.chunk_ids.append(chunk_id)
        
        self.corpus = [chunk[1] for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"BM25 index built with {len(self.corpus)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25"""
        # Convert to lowercase and split on whitespace
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]
    
    def rerank(self, query: str, vector_results: List[Dict]) -> List[Dict]:
        """Rerank vector results using hybrid scoring"""
        if not vector_results or not self.bm25:
            return vector_results
        
        # Get BM25 scores for the query
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Create mapping from chunk_id to BM25 score
        chunk_id_to_bm25 = {}
        for i, chunk_id in enumerate(self.chunk_ids):
            chunk_id_to_bm25[chunk_id] = bm25_scores[i]
        
        # Extract vector scores and BM25 scores for reranking
        vector_scores = [result['vector_score'] for result in vector_results]
        bm25_scores_subset = [chunk_id_to_bm25.get(result['id'], 0.0) for result in vector_results]
        
        # Normalize both score types
        normalized_vector_scores = self._normalize_scores(vector_scores)
        normalized_bm25_scores = self._normalize_scores(bm25_scores_subset)
        
        # Calculate hybrid scores
        hybrid_results = []
        for i, result in enumerate(vector_results):
            hybrid_score = (ALPHA * normalized_vector_scores[i] + 
                          (1 - ALPHA) * normalized_bm25_scores[i])
            
            result_copy = result.copy()
            result_copy['hybrid_score'] = hybrid_score
            result_copy['bm25_score'] = bm25_scores_subset[i]
            result_copy['normalized_vector_score'] = normalized_vector_scores[i]
            result_copy['normalized_bm25_score'] = normalized_bm25_scores[i]
            
            hybrid_results.append(result_copy)
        
        # Sort by hybrid score (descending)
        hybrid_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return hybrid_results
    
    def search_with_reranking(self, query: str, vector_search, k: int = 10) -> List[Dict]:
        """Perform vector search and rerank results"""
        # Get more candidates for reranking
        candidates = vector_search.search(query, k=CANDIDATE_K)
        
        if not candidates:
            return []
        
        # Rerank candidates
        reranked_results = self.rerank(query, candidates)
        
        # Return top k results
        return reranked_results[:k]
    
    def get_keyword_matches(self, query: str, text: str) -> int:
        """Count keyword matches between query and text"""
        query_tokens = set(self._tokenize(query))
        text_tokens = set(self._tokenize(text))
        
        # Count exact matches
        exact_matches = len(query_tokens.intersection(text_tokens))
        
        # Count partial matches (substring)
        partial_matches = 0
        for query_token in query_tokens:
            for text_token in text_tokens:
                if query_token in text_token or text_token in query_token:
                    partial_matches += 1
        
        return exact_matches + partial_matches * 0.5  # Weight partial matches less


if __name__ == "__main__":
    from vector_search import VectorSearch
    
    # Test the reranker
    reranker = HybridReranker()
    vector_search = VectorSearch()
    vector_search.load_index()
    
    test_query = "What is ISO 13849-1 safety standard?"
    
    print("=== Baseline Vector Search ===")
    baseline_results = vector_search.search(test_query, k=5)
    for i, result in enumerate(baseline_results, 1):
        print(f"{i}. Vector Score: {result['vector_score']:.3f}")
        print(f"   Source: {result['title']}")
        print(f"   Text: {result['chunk_text'][:100]}...")
        print()
    
    print("=== Hybrid Reranked Results ===")
    reranked_results = reranker.search_with_reranking(test_query, vector_search, k=5)
    for i, result in enumerate(reranked_results, 1):
        print(f"{i}. Hybrid Score: {result['hybrid_score']:.3f}")
        print(f"   Vector: {result['normalized_vector_score']:.3f}, BM25: {result['normalized_bm25_score']:.3f}")
        print(f"   Source: {result['title']}")
        print(f"   Text: {result['chunk_text'][:100]}...")
        print()
