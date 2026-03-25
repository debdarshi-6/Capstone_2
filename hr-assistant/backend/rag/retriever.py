from .vector_store import vstore_manager

def get_hr_policy_retriever(k=4):
    """
    Returns the configured retriever for HR policies, guidelines and FAQs.
    """
    return vstore_manager.get_retriever(k=k)

def search_knowledge_base(query: str, k=4):
    """
    Directly searches the knowledge base via the vector store.
    Formats the output for easy reading by the LLM Agent.
    """
    print(f"Searching knowledge base for: {query}")
    retriever = get_hr_policy_retriever(k)
    docs = retriever.invoke(query)
    
    if not docs:
        return "No relevant HR policies or guidelines found."
        
    # Format the retrieved docs for the LLM
    formatted_docs = "\n\n".join(
        [f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" for doc in docs]
    )
    return formatted_docs
