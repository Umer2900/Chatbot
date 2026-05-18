from langchain_core.tools import tool
from typing import Optional
import os
import requests
from .rag import _get_retriever, _THREAD_METADATA


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> str:
    """Retrieve relevant information from the PDF document uploaded in this chat thread."""
    if not thread_id:
        return "Error: Could not determine current chat thread."

    retriever = _get_retriever(thread_id)       # tools.py Calls rag.py Function to get retriever object for this thread ID, which was stored in memory when the PDF was ingested. If no PDF was ingested for this thread, retriever will be None.
    if retriever is None:
        return "No document has been uploaded for this chat. Please upload a PDF first."

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content[:750] for doc in docs])
    filename = _THREAD_METADATA.get(str(thread_id), {}).get("filename", "PDF")

    return f"""Relevant information from the uploaded document '{filename}':\n\n{context}\n\nAnswer the user's question using the context above."""


@tool
def tool_tavily_search(query: str) -> str:
    """Search the web for current events and general information."""
    try:
        from langchain_tavily import TavilySearch
        search = TavilySearch(max_results=3)
        return str(search.invoke(query))[:2000]
    except Exception as e:
        return f"Tavily error: {str(e)}"    

@tool 
def tool_wikipedia_search(query: str) -> str:
    """Search Wikipedia for factual information about people, places, or concepts."""
    try:
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return wikipedia.invoke(query)
    except Exception as e:
        return f"Wikipedia error: {str(e)}"


@tool
def tool_arxiv_search(query: str) -> str:
    """Search for scientific papers and research on Arxiv."""
    try:
        from langchain_community.tools import ArxivQueryRun
        from langchain_community.utilities import ArxivAPIWrapper
        arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2))
        return arxiv.run(query)[:2000]
    except Exception as e:
        return f"Arxiv error: {str(e)}"


@tool
def get_stock_price(symbol: str) -> str:
    """Fetch the latest stock price for a given symbol using Alpha Vantage."""
    try:
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            return "Alpha Vantage API key not found."
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return str(response.json())
    except Exception as e:
        return f"Stock price error: {str(e)}"


tools = [tool_tavily_search, tool_wikipedia_search, tool_arxiv_search, get_stock_price, rag_tool]
