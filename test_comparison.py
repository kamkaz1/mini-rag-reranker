"""
Test and comparison tool for the Mini RAG system
"""
import requests
import json
from typing import List, Dict
import time


class RAGTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_questions = [
            "What is ISO 13849-1?",
            "What are the different performance levels in safety systems?",
            "What is the purpose of machine guarding?",
            "How does risk assessment work in machinery safety?",
            "What are the electrical safety requirements for industrial machinery?",
            "What is functional safety according to IEC 61508?",
            "What types of machine guards are available?",
            "What are the key principles of machinery safety design?"
        ]
    
    def ask_question(self, question: str, mode: str = "baseline", k: int = 10) -> Dict:
        """Ask a question to the API"""
        try:
            response = requests.post(
                f"{self.base_url}/ask",
                json={
                    "q": question,
                    "k": k,
                    "mode": mode
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def compare_modes(self, question: str, k: int = 10) -> Dict:
        """Compare baseline vs reranked results for a question"""
        print(f"\nQuestion: {question}")
        print("=" * 80)
        
        # Test baseline
        print("Testing baseline mode...")
        start_time = time.time()
        baseline_result = self.ask_question(question, "baseline", k)
        baseline_time = time.time() - start_time
        
        # Test reranked
        print("Testing reranked mode...")
        start_time = time.time()
        reranked_result = self.ask_question(question, "reranked", k)
        reranked_time = time.time() - start_time
        
        return {
            "question": question,
            "baseline": {
                "result": baseline_result,
                "time": baseline_time
            },
            "reranked": {
                "result": reranked_result,
                "time": reranked_time
            }
        }
    
    def print_comparison(self, comparison: Dict):
        """Print a formatted comparison"""
        question = comparison["question"]
        baseline = comparison["baseline"]
        reranked = comparison["reranked"]
        
        print(f"\nQuestion: {question}")
        print("=" * 80)
        
        # Baseline results
        print("\n🔍 BASELINE RESULTS:")
        print(f"Time: {baseline['time']:.3f}s")
        if "error" in baseline["result"]:
            print(f"Error: {baseline['result']['error']}")
        else:
            print(f"Answer: {baseline['result'].get('answer', 'No answer')}")
            print(f"Reason: {baseline['result'].get('reason', 'N/A')}")
            print("Top 3 contexts:")
            for i, ctx in enumerate(baseline["result"].get("contexts", [])[:3], 1):
                print(f"  {i}. Score: {ctx['score']:.3f} | {ctx['source']}")
        
        # Reranked results
        print("\n🎯 RERANKED RESULTS:")
        print(f"Time: {reranked['time']:.3f}s")
        if "error" in reranked["result"]:
            print(f"Error: {reranked['result']['error']}")
        else:
            print(f"Answer: {reranked['result'].get('answer', 'No answer')}")
            print(f"Reason: {reranked['result'].get('reason', 'N/A')}")
            print("Top 3 contexts:")
            for i, ctx in enumerate(reranked["result"].get("contexts", [])[:3], 1):
                print(f"  {i}. Score: {ctx['score']:.3f} | {ctx['source']}")
        
        # Comparison
        print("\n📊 COMPARISON:")
        if "error" not in baseline["result"] and "error" not in reranked["result"]:
            baseline_answer = baseline["result"].get("answer")
            reranked_answer = reranked["result"].get("answer")
            
            if baseline_answer and reranked_answer:
                print("✅ Both modes provided answers")
            elif baseline_answer and not reranked_answer:
                print("⚠️  Only baseline provided an answer")
            elif not baseline_answer and reranked_answer:
                print("🎯 Only reranked provided an answer")
            else:
                print("❌ Neither mode provided an answer")
            
            # Check if reranking changed the order
            baseline_contexts = baseline["result"].get("contexts", [])
            reranked_contexts = reranked["result"].get("contexts", [])
            
            if baseline_contexts and reranked_contexts:
                baseline_top = baseline_contexts[0]["source"] if baseline_contexts else "None"
                reranked_top = reranked_contexts[0]["source"] if reranked_contexts else "None"
                
                if baseline_top != reranked_top:
                    print(f"🔄 Reranking changed top result: {baseline_top} → {reranked_top}")
                else:
                    print("📍 Reranking kept same top result")
    
    def run_full_test(self):
        """Run full test suite"""
        print("🚀 Starting Mini RAG + Reranker Test Suite")
        print("=" * 80)
        
        # Check if API is running
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ API is running")
            else:
                print("❌ API health check failed")
                return
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            print("Make sure to run: uvicorn main:app --reload")
            return
        
        # Get system stats
        try:
            stats_response = requests.get(f"{self.base_url}/stats", timeout=5)
            if stats_response.status_code == 200:
                stats = stats_response.json()
                print(f"📊 System stats: {stats['total_chunks']} chunks, {stats['embedding_model']}")
        except Exception as e:
            print(f"⚠️  Could not get system stats: {e}")
        
        # Run comparisons
        results = []
        for i, question in enumerate(self.test_questions, 1):
            print(f"\n{'='*20} TEST {i}/{len(self.test_questions)} {'='*20}")
            comparison = self.compare_modes(question)
            self.print_comparison(comparison)
            results.append(comparison)
        
        # Summary
        print(f"\n{'='*20} SUMMARY {'='*20}")
        total_tests = len(results)
        baseline_answers = sum(1 for r in results if r["baseline"]["result"].get("answer"))
        reranked_answers = sum(1 for r in results if r["reranked"]["result"].get("answer"))
        
        print(f"Total questions tested: {total_tests}")
        print(f"Baseline answers: {baseline_answers}/{total_tests} ({baseline_answers/total_tests*100:.1f}%)")
        print(f"Reranked answers: {reranked_answers}/{total_tests} ({reranked_answers/total_tests*100:.1f}%)")
        
        avg_baseline_time = sum(r["baseline"]["time"] for r in results) / total_tests
        avg_reranked_time = sum(r["reranked"]["time"] for r in results) / total_tests
        
        print(f"Average baseline time: {avg_baseline_time:.3f}s")
        print(f"Average reranked time: {avg_reranked_time:.3f}s")
        
        if reranked_answers > baseline_answers:
            print("🎯 Reranker improved answer coverage!")
        elif reranked_answers == baseline_answers:
            print("📊 Reranker maintained answer coverage")
        else:
            print("⚠️  Reranker reduced answer coverage")


if __name__ == "__main__":
    tester = RAGTester()
    tester.run_full_test()
