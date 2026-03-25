import os
from langchain_community.document_loaders import PyPDFLoader

def parse_resume(file_path: str) -> str:
    """
    Parses a PDF resume and extracts the text content.
    Returns the extracted text as a single string.
    """
    if not os.path.exists(file_path):
        return f"Error: File {file_path} not found."
    
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        text = "\n".join([page.page_content for page in pages])
        return text
    except Exception as e:
        return f"Error parsing resume: {str(e)}"