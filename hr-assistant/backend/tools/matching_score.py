from langchain_core.prompts import PromptTemplate
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.models.llm_engine import get_llm

def calculate_matching_score(resume_text: str, job_description: str) -> dict:
    """
    Compares the resume against the job description to calculate an ATS matching score
    and identify missing skills/requirements.
    Returns a dict containing 'score', 'missing_skills', and 'reasoning'.
    """
    llm = get_llm()
    prompt = PromptTemplate.from_template(
        "You are an expert ATS (Applicant Tracking System). Analyze the candidate's resume against the Job Description.\n"
        "Provide a JSON response with exactly three keys:\n"
        "1. 'score': An integer from 0 to 100 representing the match percentage.\n"
        "2. 'missing_skills': A list of strings of skills required in the JD but missing from the resume.\n"
        "3. 'reasoning': A brief string explaining the score.\n\n"
        "Job Description:\n{job_desc}\n\n"
        "Resume:\n{resume}\n\n"
        "Ensure the output is ONLY valid JSON, beginning with {{ and ending with }}.\n"
    )
    chain = prompt | llm
    response = chain.invoke({"job_desc": job_description, "resume": resume_text})
    
    try:
        content = response.content if hasattr(response, 'content') else str(response)
        content = content.replace("```json", "").replace("```", "").strip()
        
        # Attempt to find json bracket boundaries if ollama outputs conversational text
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            content = content[start_idx:end_idx]
            
        data = json.loads(content)
        return data
    except Exception as e:
        return {
            "score": 0,
            "missing_skills": [],
            "reasoning": f"Parsing failed: {e}. Output: {response}"
        }