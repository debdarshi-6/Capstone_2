import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# Define path for ChromaDB persistence at the root of the project
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")

class VectorStoreManager:
    def __init__(self):
        # We use a reliable, lightweight local embedding model from HuggingFace
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db_dir = DB_DIR
        
        # Initialize ChromaDB persistent client
        self.vector_store = Chroma(
            persist_directory=self.db_dir, 
            embedding_function=self.embedding_function,
            collection_name="hr_knowledge_base"
        )

    def load_and_index_documents(self, data_dir: str):
        """Loads PDFs or text files from the data directory and indexes them."""
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            return 0
            
        # Using DirectoryLoader to flexibly load different file types
        text_loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents = text_loader.load()
        
        try:
            pdf_loader = DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
            documents.extend(pdf_loader.load())
        except Exception as e:
            print(f"Error loading PDFs: {e}")

        if not documents:
            return 0
            
        # Strategy: RecursiveCharacterTextSplitter for logical semantic chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def get_retriever(self, k=4):
        return self.vector_store.as_retriever(search_kwargs={"k": k})

# Global singleton instance
vstore_manager = VectorStoreManager()
