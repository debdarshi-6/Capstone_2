import time
import requests
import asyncio
import aiohttp

API_URL = "http://127.0.0.1:8000"

# --- C. Latency Tests ---
def test_latency():
    print("--- Running Latency Tests ---")
    
    # 1. Retrieval Time (RAG Endpoint)
    start_time = time.time()
    res = requests.post(f"{API_URL}/rag/query", json={"query": "company leave policy", "k": 3})
    retrieval_time = time.time() - start_time
    print(f"Retrieval Time (ChromaDB): {retrieval_time:.4f} seconds")
    
    # 2. End-to-End LLM Response Time (Chat Endpoint)
    start_time = time.time()
    res = requests.post(f"{API_URL}/chat", json={"message": "What is the policy for vacation?", "thread_id": "latency_test"})
    llm_time = time.time() - start_time
    print(f"End-to-End LLM Chat Response Time: {llm_time:.4f} seconds")

# --- D. Stress Tests ---
async def fetch_chat(session, message, i):
    start = time.time()
    async with session.post(f"{API_URL}/chat", json={"message": message, "thread_id": f"stress_test_{i}"}) as response:
        await response.json()
        duration = time.time() - start
        return duration

async def run_stress_test(num_concurrent=5):
    print(f"\n--- Running Stress Test ({num_concurrent} Parallel Requests) ---")
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(num_concurrent):
            tasks.append(fetch_chat(session, f"Tell me about candidate {i}", i))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successes = [r for r in results if isinstance(r, float)]
        print(f"Successful Parallel Requests: {len(successes)}/{num_concurrent}")
        if successes:
            print(f"Average Parallel Latency: {sum(successes)/len(successes):.4f} seconds")

if __name__ == "__main__":
    test_latency()
    
    # Ensure the server is running natively before executing the stress test
    print("\nNote: Make sure your FastAPI backend is running via `python backend/main.py` before running this test.")
    try:
        asyncio.run(run_stress_test(5))
    except Exception as e:
        print("Stress test failed to connect to the server. Is it running?")
