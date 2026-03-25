from langchain_ollama import ChatOllama

def get_llm():
    """
    Returns the initialized ChatOllama instance running the Llama 3.2 model.
    """
    # Using the smaller, faster llama3.2 model to reduce lag
    # Temperature set low for more deterministic output in HR matching
    llm = ChatOllama(model="llama3.1:8b", temperature=0.1)
    return llm
