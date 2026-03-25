from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

from backend.rag.retriever import search_knowledge_base
from backend.tools.matching_score import ATSScorer
from backend.tools.ranking import get_best_candidate
from backend.tools.skill_extractor import LLMSkillExtractor

# Initialize LLM
llm = ChatOllama(model="llama3.1:8b")

@tool
def search_policies(query: str) -> str:
    """Useful to search HR policies or general knowledge base for the company."""
    results = search_knowledge_base(query, k=3)
    return "\n\n".join(results) if results else "No policies found."

@tool
def score_resumes(job_description: str) -> str:
    """Useful to get ATS scores for all candidates based on a job description."""
    scorer = ATSScorer()
    results = scorer.compute_ats_scores(job_description)
    return str(results)

@tool
def rank_candidates(job_description: str) -> str:
    """Useful to find the single best candidate and compare top candidates for a job description."""
    return str(get_best_candidate(job_description))

@tool
def extract_candidate_skills(query: str) -> str:
    """Useful to extract structured JSON skills and experience for a specific candidate from their resume."""
    results = search_knowledge_base(query, k=3)
    if not results:
        return "No candidate found."
    extractor = LLMSkillExtractor()
    extracted_json = extractor._extract_with_llm("\n\n".join(results))
    return json.dumps(extracted_json) if isinstance(extracted_json, dict) else str(extracted_json)

tools = [search_policies, score_resumes, rank_candidates, extract_candidate_skills]

import json

system_prompt = "You are a helpful HR Assistant. You can search company policies, extract structured candidate skills/experience, score candidate resumes, and rank the best candidates based on a job description. Use your tools to answer the user's questions."

tools_map = {
    "search_policies": search_policies,
    "score_resumes": score_resumes,
    "rank_candidates": rank_candidates,
    "extract_candidate_skills": extract_candidate_skills
}

# Bind tools so Llama 3.1 knows their schemas
llm_with_tools = llm.bind_tools(tools)

def run_agent(message: str, thread_id: str) -> str:
    """
    Main entry point for conversational agent.
    Mocks memory handling with thread_id for now.
    """
    messages = [
        ("system", system_prompt),
        ("user", message)
    ]
    
    from langchain_core.messages import ToolMessage

    for _ in range(5): # max 5 steps
        response = llm_with_tools.invoke(messages)
        content = response.content
        
        # 1. Check for standard parsed tool calls
        if hasattr(response, 'tool_calls') and response.tool_calls:
            messages.append(response) # Append the AI's tool call message
            for tc in response.tool_calls:
                func_name = tc["name"]
                args = tc["args"]
                if func_name in tools_map:
                    tool_res = tools_map[func_name].invoke(args)
                    messages.append(ToolMessage(name=func_name, content=str(tool_res), tool_call_id=tc.get("id", "1")))
            continue
            
        # 2. Fallback check for raw <|python_tag|> output (which breaks LangGraph)
        if isinstance(content, str) and "<|python_tag|>" in content:
            try:
                json_str = content.split("<|python_tag|>")[1].strip()
                call_data = json.loads(json_str)
                func_name = call_data.get("name")
                args = call_data.get("parameters", {})
                
                if func_name in tools_map:
                    tool_res = tools_map[func_name].invoke(args)
                    messages.append(("assistant", content))
                    messages.append(("user", f"Tool output from {func_name}:\n{tool_res}\n\nNow provide the final answer to the user."))
                    continue
            except Exception as e:
                messages.append(("assistant", content))
                messages.append(("user", f"Tool call failed to parse: {e}. Please retry or answer directly."))
                continue
                
        # 3. If no tool calls, it's the final answer
        return content
        
    return "Agent ran into an error or looping state."
