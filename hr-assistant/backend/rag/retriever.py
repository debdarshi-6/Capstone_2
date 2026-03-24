from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGService:
    def __init__(self, vectorstore):
        # 1. Setup Retriever
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        
        # 2. Setup LLM
        self.llm = Ollama(model="mistral")
        
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
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain.invoke(user_query) 