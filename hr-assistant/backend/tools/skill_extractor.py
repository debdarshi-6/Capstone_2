from langchain_community.llms import Ollama
import json
import json

import re

class LLMSkillExtractor:
    def __init__(self):
        self.llm = Ollama(model="llama3.1:8b")

    def _extract_with_llm(self, text):
        prompt = f"""
        Extract structured information from this resume.

        Return ONLY valid JSON:

        {{
            "name": "",
            "skills": [],
            "experience": "",
            "job_roles": []
        }}

        Resume:
        {text[:4000]}
        """

        response = self.llm.invoke(prompt)

        try:
            return json.loads(response)
        except:
            return self._safe_json_extract(response)

    def _safe_json_extract(self, text):
        # Remove markdown code blocks if present
        text = re.sub(r'```(?:json)?\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL).strip()
        
        # Try to find the outermost brackets assuming the LLM might have added conversational filler
        try:
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx+1]
                return json.loads(json_str)
        except Exception:
            pass
            
        return {}