from backend.tools.ranking import get_best_candidate

jd_text = """
Looking for a software engineer with experience in Python, FastAPI, and microservices.
"""

results = get_best_candidate(jd_text, top_k=3)

print(results)