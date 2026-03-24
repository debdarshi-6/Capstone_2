from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

from backend.tools.resume_parser import ResumeParser
from backend.rag.vector_store import VectorStoreManager
from backend.tools.matching_score import ATSScorer
from backend.tools.ranking import get_best_candidate

app = FastAPI()

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

parser = ResumeParser()
vs_manager = VectorStoreManager()
ats = ATSScorer()


# ------------------ 1. UPLOAD + PARSE + STORE ------------------
@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)):

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

    return parsed


# ------------------ 2. ATS SCORING ------------------
@app.post("/ats-score/")
async def ats_score(jd_text: str = Form(...)):
    results = ats.compute_ats_scores(jd_text)
    return results


# ------------------ 3. BEST CANDIDATE ------------------
@app.post("/best-candidate/")
async def best_candidate(jd_text: str = Form(...)):
    result = get_best_candidate(jd_text)
    return {"result": result}


# ------------------ 4. SCREENING (COMBINED) ------------------
@app.post("/screening/")
async def screening(file: UploadFile = File(...)):

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    parsed = parser.parse(file_path)

    # Convert parsed resume to text
    resume_text = str(parsed)

    # Dummy JD (you can pass from frontend later)
    jd_text = "Python developer with ML and FastAPI experience"

    ats_results = ats.compute_ats_scores(jd_text)

    best = get_best_candidate(jd_text)

    return {
        "parsed_resume": parsed,
        "ats_results": ats_results,
        "best_candidate": best
    }