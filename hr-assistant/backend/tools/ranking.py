from backend.rag.vector_store import VectorStoreManager
from langchain_ollama import OllamaLLM


def get_best_candidate(jd_text, top_k=3):

    # Step 1: Load DB
    vs = VectorStoreManager()
    vectorstore = vs.load_vector_store()

    # Step 2: Retrieve top resumes
    docs = vectorstore.similarity_search(jd_text, k=top_k)

    # Step 3: Prepare context for LLM
    resumes_text = ""
    for i, doc in enumerate(docs):
        resumes_text += f"""
        Candidate {i+1}:
        Name: {doc.metadata.get("candidate")}
        Resume:
        {doc.page_content}
        ----------------------
        """

    # Step 4: LLM
    llm = OllamaLLM(model="llama3.1:8b")

    prompt = f"""
    You are an expert HR recruiter.

    Job Description:
    {jd_text}

    Candidates:
    {resumes_text}

    Task:
    1. Compare candidates
    2. Select BEST candidate
    3. Explain WHY
    4. Give ranking

    Output format:
    - Best Candidate:
    - Reason:
    - Ranking:
    """

    response = llm.invoke(prompt)

    return response