import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.rag.retriever import get_hr_policy_retriever

def evaluate_retrieval(queries, ground_truths, k=4):
    """
    Evaluates the retrieval pipeline using Hit Rate and Recall approximation.
    """
    retriever = get_hr_policy_retriever(k)
    hits = 0
    
    print("Starting RAG Evaluation...")
    for query, truth in zip(queries, ground_truths):
        docs = retriever.invoke(query)
        retrieved_content = " ".join([d.page_content for d in docs])
        
        # Simple sub-string matching for Hit Rate. 
        # For precision, more advanced token overlaps are recommended.
        if truth.lower() in retrieved_content.lower():
            hits += 1
            print(f"✅ Hit for query: '{query}'")
        else:
            print(f"❌ Miss for query: '{query}'")
            
    hit_rate = hits / len(queries) if queries else 0
    print(f"\n--- Results ---")
    print(f"Top-{k} Hit Rate: {hit_rate * 100:.2f}%")
    return hit_rate

if __name__ == "__main__":
    # Example usage:
    test_queries = ["What is the remote work policy?"]
    test_truths = ["Employees may work remotely up to 3 days a week."]
    evaluate_retrieval(test_queries, test_truths)
