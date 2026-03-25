import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil

from backend.models.agent import run_agent
from backend.rag.retriever import search_knowledge_base
from backend.rag.vector_store import vstore_manager

app = FastAPI(title="HR Recruitment Assistant API", version="1.0")

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default"

class RagQueryRequest(BaseModel):
    query: str
    k: int = 4

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

@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload a resume to the server for processing by the agent.
    Returns the file path.
    """
    try:
        # Create a temp directory for uploads if not exists
        upload_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "temp_uploads")
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {"filename": file.filename, "file_path": file_path, "message": "Upload successful"}
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
