from backend.rag.vector_store import VectorStoreManager
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaEmbeddings


class ATSScorer:

    def __init__(self):
        self.vs = VectorStoreManager()
        self.vectorstore = self.vs.load_vector_store()
        self.embeddings = OllamaEmbeddings(model="llama3.1:8b")

    def compute_ats_scores(self, jd_text, top_k=3):

        # Step 1: Retrieve resumes
        docs = self.vectorstore.similarity_search(jd_text, k=top_k)

        if not docs:
            return []

        # Step 2: Embed JD once
        jd_embedding = self.embeddings.embed_query(jd_text)

        results = []
        seen = set()

        for doc in docs:
            candidate = doc.metadata.get("candidate")

            # Deduplicate
            if candidate in seen:
                continue
            seen.add(candidate)

            resume_text = doc.page_content

            # Step 3: Embed resume
            resume_embedding = self.embeddings.embed_query(resume_text)

            # Step 4: Cosine similarity (raw score)
            similarity = cosine_similarity(
                [jd_embedding],
                [resume_embedding]
            )[0][0]

            # Normalize safely to 0–1 range (optional but cleaner)
            similarity = max(0.0, min(1.0, similarity))

            # Step 5: ATS score (keep both raw + percentage view)
            ats_score = round(similarity, 4)        # raw (0–1)
            ats_percent = round(similarity * 100, 2)  # human-readable

            results.append({
                "candidate": candidate,
                "ats_score": ats_score,          # normalized score
                "ats_percent": ats_percent,      # percentage view
                "resume": resume_text
            })

        # Step 6: Sort by similarity (not multiplied value)
        results = sorted(results, key=lambda x: x["ats_score"], reverse=True)

        return results