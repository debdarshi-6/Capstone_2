import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from backend.models.llm_engine import get_llm
from backend.rag.retriever import search_knowledge_base
from backend.tools.resume_parser import parse_resume
from backend.tools.skill_extractor import extract_skills
from backend.tools.matching_score import calculate_matching_score
from backend.tools.validation import validate_job_description, validate_resume

# Defined Tools
@tool
def retrieve_hr_policy(query: str) -> str:
    """Use this tool to search the HR knowledge base for company policies, guidelines, and FAQs."""
    return search_knowledge_base(query)

@tool
def analyze_candidate(resume_path: str, job_description: str) -> str:
    """Use this tool to calculate a candidate's ATS matching score. 
    You must provide the local file path to the candidate's PDF resume and the text of the job description."""
    
    if not validate_job_description(job_description):
        return "Error: Job description provided is too short. Please ask the user for a more detailed one."
        
    # 1. Parse PDF
    resume_text = parse_resume(resume_path)
    if "Error parsing resume" in resume_text:
        return resume_text
        
    if not validate_resume(resume_text):
        return "Error: Resume text is invalid or too short. Check the file content."
        
    # 2. Extract Skills
    skills = extract_skills(resume_text)
    
    # 3. Calculate match
    match_result = calculate_matching_score(resume_text, job_description)
    
    # 4. Format output
    result = f"Candidate Match Score: {match_result.get('score', 0)}%\n"
    result += f"Identified Skills: {', '.join(skills)}\n"
    result += f"Missing Skills: {', '.join(match_result.get('missing_skills', []))}\n"
    result += f"Reasoning: {match_result.get('reasoning', '')}"
    
    return result

tools = [retrieve_hr_policy, analyze_candidate]

# Initialize LLM and Checkpointer (Memory)
llm = get_llm()
memory = MemorySaver()

system_prompt = (
    "You are an expert HR Recruitment Assistant. "
    "You can help screen resumes, match candidates to job descriptions, rank multiple candidates, and answer questions about company policies. "
    "Use the tools provided to access documents or analyze files. "
    "If the user asks you to evaluate multiple candidates, MUST use the analyze_candidate tool for EACH resume path individually. "
    "After analyzing all given candidates, summarize their pros/cons and clearly rank them from best fit to worst fit based on the Job Description. "
    "Maintain a professional, helpful, and objective tone."
)

# LangGraph ReAct Agent
hr_agent_executor = create_react_agent(
    model=llm,
    tools=tools,
    checkpointer=memory,
    prompt=system_prompt
)

def run_agent(user_input: str, thread_id: str = "default_thread") -> str:
    """Invoke the agent and return the final string response."""
    config = {"configurable": {"thread_id": thread_id}}
    result = hr_agent_executor.invoke(
        {"messages": [("user", user_input)]},
        config=config
    )
    # The result contains a list of messages. We want the last AI response.
    return result["messages"][-1].content
