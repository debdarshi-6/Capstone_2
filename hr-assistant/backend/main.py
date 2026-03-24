from backend.tools.resume_parser import ResumeParser
from backend.rag.vector_store import VectorStoreManager
from backend.rag.retriever import RAGService


def process_uploaded_pdf(pdf_path, query):

    parser = ResumeParser()

    # Step 1: Parse resume
    parsed_data = parser.parse(pdf_path)

    # Step 2: Convert JSON → TEXT
    text = f"""
    Name: {parsed_data.get("name")}
    Email: {parsed_data.get("email")}
    Phone: {parsed_data.get("phone")}
    Skills: {", ".join(parsed_data.get("skills", []))}
    Experience: {parsed_data.get("experience")}
    Job Roles: {", ".join(parsed_data.get("job_roles", []))}
    """

    vs = VectorStoreManager()

    # ✅ ADD resume to existing DB (IMPORTANT FIX)
    vs.add_resume(
        text=text,
        candidate_name=parsed_data.get("name"),
        file_path=pdf_path
    )

    # Step 3: Load DB for RAG
    vectorstore = vs.load_vector_store()

    # Step 4: RAG query
    rag = RAGService(vectorstore)
    response = rag.query(query)

    return {
        "parsed_data": parsed_data,
        "rag_response": response
    }