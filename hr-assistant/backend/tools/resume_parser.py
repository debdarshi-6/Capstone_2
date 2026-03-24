from langchain_community.document_loaders import PyPDFLoader

def parse_pdf(file_path) -> str:
    """
        Reads a PDF file and extracts text using LangChain PyPDFLoader.
    """
    print("----Parsing PDF file using PyPDFLoader----")
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])

        return text.strip()

    except Exception as e:
        return str(e) 