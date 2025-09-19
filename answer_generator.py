"""
Answer generation module with citation support and uncertainty handling
"""
import re
from typing import List, Dict, Optional, Tuple
from config import SIMILARITY_THRESHOLD


class AnswerGenerator:
    def __init__(self):
        self.threshold = SIMILARITY_THRESHOLD
    
    def generate_answer(self, query: str, contexts: List[Dict], reranker_used: bool = False) -> Tuple[Optional[str], str]:
        """
        Generate an answer from contexts with citations
        
        Returns:
            Tuple of (answer, reason)
            - answer: Generated answer string or None if uncertain
            - reason: Explanation for the answer or abstention
        """
        if not contexts:
            return None, "No relevant contexts found"
        
        # Check if top result meets confidence threshold
        top_score = contexts[0].get('hybrid_score', contexts[0].get('vector_score', 0))
        
        if top_score < self.threshold:
            return None, f"Top result score ({top_score:.3f}) below confidence threshold ({self.threshold})"
        
        # Extract relevant information from contexts
        answer_parts = []
        citations = []
        
        for i, context in enumerate(contexts[:3]):  # Use top 3 contexts
            score = context.get('hybrid_score', context.get('vector_score', 0))
            
            # Only include contexts with reasonable scores
            if score < self.threshold * 0.7:  # Lower threshold for supporting contexts
                continue
            
            # Extract relevant sentences from the chunk
            relevant_sentences = self._extract_relevant_sentences(query, context['chunk_text'])
            
            if relevant_sentences:
                answer_parts.extend(relevant_sentences)
                citations.append({
                    'source': context['title'],
                    'url': context['url'],
                    'score': score,
                    'text': context['chunk_text'][:200] + "..." if len(context['chunk_text']) > 200 else context['chunk_text']
                })
        
        if not answer_parts:
            return None, "No relevant information found in contexts"
        
        # Combine answer parts and add citations
        answer = self._combine_answer_parts(answer_parts, citations)
        
        return answer, f"Answer generated from {len(citations)} sources with top score {top_score:.3f}"
    
    def _extract_relevant_sentences(self, query: str, text: str) -> List[str]:
        """Extract sentences most relevant to the query"""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Score sentences based on keyword overlap
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        scored_sentences = []
        
        for sentence in sentences:
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            overlap = len(query_words.intersection(sentence_words))
            scored_sentences.append((sentence, overlap))
        
        # Sort by relevance and return top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        relevant_sentences = []
        for sentence, score in scored_sentences[:2]:  # Top 2 sentences
            if score > 0 and len(sentence) > 20:  # Minimum relevance and length
                relevant_sentences.append(sentence)
        
        return relevant_sentences
    
    def _combine_answer_parts(self, answer_parts: List[str], citations: List[Dict]) -> str:
        """Combine answer parts into a coherent response with citations"""
        # Remove duplicates while preserving order
        seen = set()
        unique_parts = []
        for part in answer_parts:
            if part not in seen:
                seen.add(part)
                unique_parts.append(part)
        
        # Combine parts
        answer = " ".join(unique_parts)
        
        # Add citations
        if citations:
            citation_refs = []
            for i, citation in enumerate(citations, 1):
                citation_refs.append(f"[{i}] {citation['source']}")
            
            answer += f"\n\nSources: {', '.join(citation_refs)}"
        
        return answer
    
    def format_response(self, query: str, contexts: List[Dict], reranker_used: bool = False) -> Dict:
        """Format the complete response"""
        answer, reason = self.generate_answer(query, contexts, reranker_used)
        
        # Format contexts for response
        formatted_contexts = []
        for context in contexts:
            formatted_contexts.append({
                'text': context['chunk_text'],
                'score': context.get('hybrid_score', context.get('vector_score', 0)),
                'source': context['title'],
                'url': context['url']
            })
        
        return {
            'answer': answer,
            'contexts': formatted_contexts,
            'reranker_used': reranker_used,
            'reason': reason,
            'query': query
        }


if __name__ == "__main__":
    # Test the answer generator
    generator = AnswerGenerator()
    
    # Sample test data
    test_contexts = [
        {
            'chunk_text': 'ISO 13849-1 is an international standard that specifies safety requirements for safety-related parts of control systems. It provides guidance on the design and integration of safety-related parts of control systems.',
            'title': 'ISO 13849-1:2015 - Safety of machinery',
            'url': 'https://www.iso.org/standard/69883.html',
            'vector_score': 0.85
        },
        {
            'chunk_text': 'The standard defines performance levels (PL) from PLa to PLe, where PLe represents the highest level of safety integrity.',
            'title': 'ISO 13849-1:2015 - Safety of machinery',
            'url': 'https://www.iso.org/standard/69883.html',
            'vector_score': 0.75
        }
    ]
    
    test_query = "What is ISO 13849-1?"
    
    response = generator.format_response(test_query, test_contexts, reranker_used=False)
    
    print("Query:", response['query'])
    print("Answer:", response['answer'])
    print("Reason:", response['reason'])
    print("Reranker used:", response['reranker_used'])
    print("\nContexts:")
    for i, ctx in enumerate(response['contexts'], 1):
        print(f"{i}. Score: {ctx['score']:.3f}")
        print(f"   Source: {ctx['source']}")
        print(f"   Text: {ctx['text'][:100]}...")
        print()
