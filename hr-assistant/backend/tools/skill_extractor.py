from langchain_core.prompts import PromptTemplate
import sys
import os

# Ensure backend imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from backend.models.llm_engine import get_llm

def extract_skills(text: str) -> list[str]:
    """
    Extracts a list of skills from the provided resume text using the LLM.
    """
    llm = get_llm()
    prompt = PromptTemplate.from_template(
        "You are an expert HR AI Assistant. Extract a concise list of professional skills from the following resume text. Return ONLY a comma-separated list of skills.\n\nResume Text:\n{text}\n\nSkills:"
    )
    chain = prompt | llm
    response = chain.invoke({"text": text})
    
    # Process the output
    if hasattr(response, 'content'):
        output_text = response.content
    else:
        output_text = str(response)
        
    skills = [s.strip() for s in output_text.split(",") if s.strip()]
    return skills