from langchain_ollama import ChatOllama, OllamaEmbeddings
from .config import OLLAMA_MODEL, EMBEDDING_MODEL
import os

# LLM
llm = ChatOllama( 
    model=OLLAMA_MODEL,  
    base_url="https://ollama.com", 
    client_kwargs={ 
        "headers": { 
            "Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}" 
        } 
    }, 
    temperature=0.3
    # streaming=True 
)  

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)


__all__ = ["llm", "embeddings"]