from langchain_community.llms import Ollama
import json
from pydantic import BaseModel
from typing import List, Optional


class CandidateInfo(BaseModel):
    skills: List[str]
    experience: Optional[str] = None
    location: Optional[str] = None
    
from langchain_community.llms import Ollama
import json

class LLMSkillExtractor:
    def __init__(self):
        self.llm = Ollama(model="mistral")

    def extract(self, text: str) -> CandidateInfo:
        prompt = f"""
        You are an AI HR assistant.

        Extract the following details:

        1. Skills (technical only)
        2. Experience (number only)
        3. Location with constraint (VERY IMPORTANT)

        Location Rules:
        - If location is mandatory → return: "City (mandatory)"
        - If preferred → return: "City (preferred)"
        - If remote → return: "Remote"
        - If just location mentioned → return: "City"
        - If nothing → return null

        Return ONLY JSON:

        {{
        "skills": ["skill1", "skill2"],
        "experience": number or null,
        "location": "formatted string or null"
        }}

        Text:
        {text}
        """

        response = self.llm.invoke(prompt)

        # 🔥 Clean JSON
        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            clean_json = response[json_start:json_end]

            data = json.loads(clean_json)
        except:
            data = {
                "skills": [],
                "experience": None,
                "location": None
            }

        return CandidateInfo(**data)