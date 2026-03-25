import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil

from backend.models.agent import run_agent
from backend.rag.retriever import search_knowledge_base
from backend.rag.vector_store import vstore_manager, VectorStoreManager
from backend.tools.resume_parser import ResumeParser
from backend.tools.matching_score import ATSScorer
from backend.tools.ranking import get_best_candidate

app = FastAPI(title="HR Recruitment Assistant API", version="1.0")

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use absolute path for uploads inside hr-assistant/data/temp_uploads
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize instances
parser = ResumeParser()
vs_manager = VectorStoreManager()
ats = ATSScorer()

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"

class RagQueryRequest(BaseModel):
    query: str
    k: int = 4

# ------------------ CHAT & RAG (EXISTING) ------------------

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Main entry point for conversational agent.
    Maintains memory via thread_id.
    """
    try:
        response = run_agent(request.message, request.thread_id)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/query")
async def rag_query_endpoint(request: RagQueryRequest):
    """
    Direct endpoint to query the underlying Vector Database without triggering agent tools.
    """
    try:
        results = search_knowledge_base(request.query, request.k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/ingest")
async def ingest_documents(data_dir: str):
    """
    Trigger ingestion of new HR policies into ChromaDB.
    """
    try:
        count = vstore_manager.load_and_index_documents(data_dir)
        return {"message": f"Successfully ingested {count} document chunks."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ NEW INTEGRATED ENDPOINTS ------------------

# ------------------ 1. UPLOAD + PARSE + STORE ------------------
@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parse resume
        parsed = parser.parse(file_path)

        # Store in vector DB
        vs_manager.add_resume(
            text=str(parsed),
            candidate_name=parsed.get("name"),
            file_path=file_path
        )

        parsed["file_path"] = file_path
        return parsed
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ 2. ATS SCORING ------------------
@app.post("/ats-score/")
async def ats_score(jd_text: str = Form(...)):
    try:
        results = ats.compute_ats_scores(jd_text)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ 3. BEST CANDIDATE ------------------
@app.post("/best-candidate/")
async def best_candidate(jd_text: str = Form(...)):
    try:
        result = get_best_candidate(jd_text)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ 4. SCREENING (COMBINED) ------------------
from typing import List
from langchain_ollama import OllamaLLM

@app.post("/screening/")
async def screening(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        parsed = parser.parse(file_path)

        # Dummy JD (you can pass from frontend later)
        jd_text = "Python developer with ML and FastAPI experience"

        ats_results = ats.compute_ats_scores(jd_text)
        best = get_best_candidate(jd_text)

        return {
            "parsed_resume": parsed,
            "ats_results": ats_results,
            "best_candidate": best
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------ 5. BATCH SCREENING ------------------
@app.post("/batch-screening/")
async def batch_screening(files: List[UploadFile] = File(...), jd_text: str = Form(...)):
    try:
        parsed_resumes = []
        resumes_text = ""

        llm = OllamaLLM(model="llama3.1:8b")

        for i, file in enumerate(files):
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            parsed = parser.parse(file_path)
            
            # Store in DB so Chatbot can use it later
            vs_manager.add_resume(
                text=str(parsed),
                candidate_name=parsed.get("name"),
                file_path=file_path
            )

            parsed_resumes.append(parsed)

            name = parsed.get("name") or f"Candidate {i+1}"
            resumes_text += f"""
            Candidate {i+1}:
            Name: {name}
            Resume Data:
            {str(parsed)}
            ----------------------
            """

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
        best_result = llm.invoke(prompt)

        return {
            "result": best_result,
            "parsed_resumes": parsed_resumes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)