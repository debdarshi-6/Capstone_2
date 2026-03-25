import os
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFDirectoryLoader


class VectorStoreManager:
    def __init__(self, persist_dir="backend/rag/chroma_db"):
        self.persist_dir = persist_dir

        # ✅ Correct embedding model
        self.embedding = OllamaEmbeddings(model="llama3.1:8b")

        # ✅ Text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )

    # ---------------- LOAD EXISTING DB ----------------
    def load_vector_store(self):
        """
        Load existing vector database
        """
        return Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding
        )

    # ---------------- ADD RESUME ----------------
    def add_resume(self, text, candidate_name, file_path):
        """
        Add a parsed resume into vector DB
        """

        # Create Document with metadata
        doc = Document(
            page_content=text,
            metadata={
                "candidate": candidate_name,
                "file": file_path
            }
        )

        # Split into chunks
        chunks = self.splitter.split_documents([doc])

        # Load DB
        vectorstore = self.load_vector_store()

        # Add to DB (IMPORTANT: no overwrite)
        vectorstore.add_documents(chunks)

        return "Resume stored successfully"

    # ---------------- LOAD DOCUMENTS ----------------
    def load_and_index_documents(self, data_dir):
        """
        Load policies/documents from a directory and index them into Chroma.
        Returns the number of chunks added.
        """
        if not os.path.exists(data_dir):
            return 0
            
        loader = PyPDFDirectoryLoader(data_dir)
        docs = loader.load()
        
        if not docs:
            return 0
            
        chunks = self.splitter.split_documents(docs)
        
        vectorstore = self.load_vector_store()
        vectorstore.add_documents(chunks)
        
        return len(chunks)

# Create singleton instance
vstore_manager = VectorStoreManager()