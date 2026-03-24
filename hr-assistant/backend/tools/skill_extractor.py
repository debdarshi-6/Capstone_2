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
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {} 