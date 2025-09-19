"""
PDF processing module for extracting text and creating chunks
"""
import os
import json
import re
import sqlite3
from typing import List, Dict, Tuple
import PyPDF2
from config import CHUNK_SIZE, CHUNK_OVERLAP, DATABASE_PATH, PDF_DIR, SOURCES_FILE


class PDFProcessor:
    def __init__(self):
        self.sources = self._load_sources()
        self.setup_database()
    
    def _load_sources(self) -> Dict[str, Dict]:
        """Load sources.json and create filename mapping"""
        with open(SOURCES_FILE, 'r') as f:
            data = json.load(f)
        
        sources_map = {}
        for source in data['sources']:
            sources_map[source['filename']] = {
                'title': source['title'],
                'url': source['url']
            }
        return sources_map
    
    def setup_database(self):
        """Create SQLite database and tables"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                word_count INTEGER NOT NULL,
                title TEXT NOT NULL,
                url TEXT NOT NULL
            )
        ''')
        
        # Create FTS virtual table for full-text search
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_text,
                title,
                content='chunks',
                content_rowid='id'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        return text.strip()
    
    def split_into_chunks(self, text: str, source_file: str) -> List[Dict]:
        """Split text into chunks of specified size"""
        words = text.split()
        chunks = []
        chunk_index = 0
        
        i = 0
        while i < len(words):
            # Take a chunk of words
            chunk_words = words[i:i + CHUNK_SIZE]
            chunk_text = ' '.join(chunk_words)
            
            # Try to end at sentence boundary
            if i + CHUNK_SIZE < len(words):
                # Look for sentence endings in the last 50 words
                for j in range(len(chunk_words) - 1, max(0, len(chunk_words) - 50), -1):
                    if chunk_words[j].endswith(('.', '!', '?')):
                        chunk_text = ' '.join(chunk_words[:j+1])
                        i += j + 1
                        break
                else:
                    i += CHUNK_SIZE - CHUNK_OVERLAP
            else:
                i += CHUNK_SIZE
            
            if chunk_text.strip():
                chunks.append({
                    'source_file': source_file,
                    'chunk_text': self.clean_text(chunk_text),
                    'chunk_index': chunk_index,
                    'word_count': len(chunk_text.split()),
                    'title': self.sources.get(source_file, {}).get('title', 'Unknown'),
                    'url': self.sources.get(source_file, {}).get('url', '')
                })
                chunk_index += 1
        
        return chunks
    
    def store_chunks(self, chunks: List[Dict]):
        """Store chunks in SQLite database"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        for chunk in chunks:
            cursor.execute('''
                INSERT INTO chunks (source_file, chunk_text, chunk_index, word_count, title, url)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                chunk['source_file'],
                chunk['chunk_text'],
                chunk['chunk_index'],
                chunk['word_count'],
                chunk['title'],
                chunk['url']
            ))
        
        conn.commit()
        conn.close()
    
    def process_pdf(self, pdf_path: str) -> int:
        """Process a single PDF file and return number of chunks created"""
        source_file = os.path.basename(pdf_path)
        
        if source_file not in self.sources:
            print(f"Warning: {source_file} not found in sources.json")
            return 0
        
        print(f"Processing {source_file}...")
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            print(f"Warning: No text extracted from {source_file}")
            return 0
        
        chunks = self.split_into_chunks(text, source_file)
        self.store_chunks(chunks)
        
        print(f"Created {len(chunks)} chunks from {source_file}")
        return len(chunks)
    
    def process_all_pdfs(self, pdf_directory: str = PDF_DIR) -> int:
        """Process all PDF files in the directory"""
        if not os.path.exists(pdf_directory):
            print(f"PDF directory {pdf_directory} does not exist")
            return 0
        
        total_chunks = 0
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            chunks_count = self.process_pdf(pdf_path)
            total_chunks += chunks_count
        
        # Update FTS index
        self._update_fts_index()
        
        print(f"Total chunks created: {total_chunks}")
        return total_chunks
    
    def _update_fts_index(self):
        """Update the FTS index after inserting new chunks"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Rebuild FTS index
        cursor.execute('INSERT INTO chunks_fts(chunks_fts) VALUES("rebuild")')
        
        conn.commit()
        conn.close()
    
    def get_chunk_count(self) -> int:
        """Get total number of chunks in database"""
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM chunks')
        count = cursor.fetchone()[0]
        conn.close()
        return count


if __name__ == "__main__":
    processor = PDFProcessor()
    print(f"Sources loaded: {len(processor.sources)}")
    print(f"Current chunks in database: {processor.get_chunk_count()}")
