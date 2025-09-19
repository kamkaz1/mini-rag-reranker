"""
Vector search module using sentence transformers and FAISS
"""
import os
import sqlite3
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import pickle
from config import EMBEDDING_MODEL, FAISS_INDEX_PATH, DATABASE_PATH, RANDOM_SEED


class VectorSearch:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.chunk_ids = []
        self.dimension = 384  # all-MiniLM-L6-v2 dimension
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.astype('float32')
    
    def create_index(self):
        """Create FAISS index from all chunks in database"""
        print("Loading chunks from database...")
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, chunk_text FROM chunks ORDER BY id')
        chunks = cursor.fetchall()
        conn.close()
        
        if not chunks:
            print("No chunks found in database")
            return
        
        print(f"Generating embeddings for {len(chunks)} chunks...")
        texts = [chunk[1] for chunk in chunks]
        embeddings = self.generate_embeddings(texts)
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product (cosine similarity)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Store chunk IDs
        self.chunk_ids = [chunk[0] for chunk in chunks]
        
        # Save index and chunk IDs
        self.save_index()
        
        print(f"FAISS index created with {self.index.ntotal} vectors")
    
    def save_index(self):
        """Save FAISS index and chunk IDs to disk"""
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        
        with open(FAISS_INDEX_PATH + '.ids', 'wb') as f:
            pickle.dump(self.chunk_ids, f)
        
        print(f"Index saved to {FAISS_INDEX_PATH}")
    
    def load_index(self):
        """Load FAISS index and chunk IDs from disk"""
        if not os.path.exists(FAISS_INDEX_PATH):
            print("FAISS index not found. Run create_index() first.")
            return False
        
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        
        with open(FAISS_INDEX_PATH + '.ids', 'rb') as f:
            self.chunk_ids = pickle.load(f)
        
        print(f"Index loaded with {self.index.ntotal} vectors")
        return True
    
    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search for similar chunks using vector similarity"""
        if self.index is None:
            if not self.load_index():
                return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding.astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Get chunk details from database
        results = []
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue
                
            chunk_id = self.chunk_ids[idx]
            cursor.execute('''
                SELECT id, source_file, chunk_text, chunk_index, title, url
                FROM chunks WHERE id = ?
            ''', (chunk_id,))
            
            chunk_data = cursor.fetchone()
            if chunk_data:
                results.append({
                    'id': chunk_data[0],
                    'source_file': chunk_data[1],
                    'chunk_text': chunk_data[2],
                    'chunk_index': chunk_data[3],
                    'title': chunk_data[4],
                    'url': chunk_data[5],
                    'vector_score': float(score)
                })
        
        conn.close()
        return results
    
    def get_chunk_by_id(self, chunk_id: int) -> Dict:
        """Get chunk details by ID"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, source_file, chunk_text, chunk_index, title, url
            FROM chunks WHERE id = ?
        ''', (chunk_id,))
        
        chunk_data = cursor.fetchone()
        conn.close()
        
        if chunk_data:
            return {
                'id': chunk_data[0],
                'source_file': chunk_data[1],
                'chunk_text': chunk_data[2],
                'chunk_index': chunk_data[3],
                'title': chunk_data[4],
                'url': chunk_data[5]
            }
        return None


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    vector_search = VectorSearch()
    
    # Create index if it doesn't exist
    if not os.path.exists(FAISS_INDEX_PATH):
        vector_search.create_index()
    else:
        vector_search.load_index()
    
    # Test search
    test_query = "What is ISO 13849-1?"
    results = vector_search.search(test_query, k=5)
    
    print(f"\nTest query: {test_query}")
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score: {result['vector_score']:.3f}")
        print(f"   Source: {result['title']}")
        print(f"   Text: {result['chunk_text'][:100]}...")
        print()
