from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from backend.rag.vector_store import vstore_manager

class RAGService:
    def __init__(self, vectorstore):
        # 1. Setup Retriever
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # 2. Setup LLM
        self.llm = Ollama(model="llama3.1:8b")
        
        # 3. Define Prompt Template
        self.template = """
        You are an HR Assistant. Use the provided context to answer the question. 
        If you don't know the answer based on the context, say you don't know.
        
        Context:
        {context}
        
        Question: 
        {question}
        
        Answer:
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)

    def query(self, user_query):
        # 4. Build the Chain (LCEL)
        setup_and_retrieval = RunnableParallel(
            context=self.retriever,
            question=RunnablePassthrough()
        )
        
        chain = setup_and_retrieval | self.prompt | self.llm | StrOutputParser()
        
        return chain.invoke(user_query)

def search_knowledge_base(query: str, k: int = 4):
    """
    Search the chroma database directly and return context strings.
    """
    vectorstore = vstore_manager.load_vector_store()
    if not vectorstore:
        return []
        
    docs = vectorstore.similarity_search(query, k=k)
    results = [doc.page_content for doc in docs]
    return results 