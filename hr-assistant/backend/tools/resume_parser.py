import os
import re
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaLLM
from backend.tools.skill_extractor import LLMSkillExtractor


class ResumeParser:
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.1:8b")
        self.skill_extractor = LLMSkillExtractor()

    def parse(self, file_path: str) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError("File not found")

        ext = file_path.split('.')[-1].lower()

        if ext == "pdf":
            text, pages = self._parse_pdf(file_path)
        else:
            raise ValueError("Unsupported file format")

        # Basic metadata (regex)
        basic_info = self._extract_basic_info(text)

        # LLM structured extraction
        structured_info = self.skill_extractor._extract_with_llm(text)

        # Final JSON
        return {
            "name": basic_info.get("name") or structured_info.get("name"),
            "email": basic_info.get("email"),
            "phone": basic_info.get("phone"),
            "skills": structured_info.get("skills", []),
            "experience": structured_info.get("experience"),
            "job_roles": structured_info.get("job_roles", []),
            "total_pages": pages
        }

    # ---------------- PDF ----------------
    def _parse_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        full_text = "\n".join([doc.page_content for doc in documents])
        return full_text.strip(), len(documents)

    # ---------------- REGEX BASIC ----------------
    def _extract_basic_info(self, text):
        return {
            "name": self._extract_name(text),
            "email": self._extract_email(text),
            "phone": self._extract_phone(text)
        }

    def _extract_email(self, text):
        match = re.search(r"[a-zA-Z0-9+_.-]+@[a-zA-Z0-9.-]+", text)
        return match.group(0) if match else None

    def _extract_phone(self, text):
        match = re.search(r"\+?\d[\d\s\-]{8,15}\d", text)
        return match.group(0) if match else None

    def _extract_name(self, text):
        lines = text.split("\n")
        for line in lines[:5]:
            if len(line.split()) <= 4 and line.replace(" ", "").isalpha():
                return line.strip()
        return None
    

    # ---------------- LLM EXTRACTION ----------------

    # def _extract_with_llm(self, text):
    #     prompt = f"""
    #     Extract structured information from this resume.

    #     Return ONLY valid JSON:

    #     {{
    #         "name": "",
    #         "skills": [],
    #         "experience": "",
    #         "job_roles": []
    #     }}

    #     Resume:
    #     {text[:4000]}
    #     """

    #     response = self.llm.invoke(prompt)

    #     try:
    #         return json.loads(response)
    #     except:
    #         return self._safe_json_extract(response)

    # def _safe_json_extract(self, text):
    #     match = re.search(r'\{.*\}', text, re.DOTALL)
    #     if match:
    #         return json.loads(match.group())
    #     return {} 