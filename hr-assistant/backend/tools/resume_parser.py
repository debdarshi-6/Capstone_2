import os
import re
from langchain_community.document_loaders import PyPDFLoader
# from docx import Document

class ResumeParser:
    def __init__(self):
        pass

    def parse(self, file_path: str) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError("File not found")

        ext = file_path.split('.')[-1].lower()

        if ext == "pdf":
            text, pages = self._parse_pdf(file_path)
        elif ext == "docx":
            text, pages = self._parse_docx(file_path)
        else:
            raise ValueError("Unsupported file format")

        metadata = self._extract_basic_info(text)

        return {
            "text": text,
            "pages": pages,
            "metadata": metadata
        }

    # ---------------- PDF USING PyPDFLoader ----------------
    def _parse_pdf(self, file_path):
        loader = PyPDFLoader(file_path)
        documents = loader.load()  # list of Document objects

        full_text = ""
        for doc in documents:
            full_text += doc.page_content + "\n"

        return full_text.strip(), len(documents)

    # ---------------- DOCX ----------------
    # def _parse_docx(self, file_path):
    #     doc = Document(file_path)
    #     full_text = "\n".join([para.text for para in doc.paragraphs])
    #     return full_text.strip(), len(doc.paragraphs)

    # ---------------- METADATA ----------------
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