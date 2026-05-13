from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from dotenv import load_dotenv
import sqlite3
import requests
import os
from datetime import datetime
load_dotenv()

# -------------------
# 1. LLM
# -------------------
# llm = ChatGroq(model="openai/gpt-oss-120b")
llm = ChatOllama( 
    model="gpt-oss:20b",  
    base_url="https://ollama.com", 
    client_kwargs={ 
        "headers": { 
            "Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}" 
        } 
    }, 
    temperature=0.3, 
)  


# -------------------
# 2. Tools
# -------------------

@tool
def tool_tavily_search(query: str) -> str:
    """Use for current events or general web search."""
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        search = TavilySearchResults(max_results=3)
        results = search.invoke(query)
        return str(results)[:2000]
    except Exception as e:
        return f"Tavily search error: {str(e)}"


@tool 
def tool_wikipedia_search(query: str) -> str:
    """Use for factual information."""
    try:
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return wikipedia.invoke(query)
    except Exception as e:
        return f"Wikipedia error: {str(e)}"


@tool
def tool_arxiv_search(query: str) -> str:
    """Use for scientific papers."""
    try:
        from langchain_community.tools import ArxivQueryRun
        from langchain_community.utilities import ArxivAPIWrapper
        arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=2))
        return arxiv.run(query)[:3000]
    except Exception as e:
        return f"Arxiv error: {str(e)}"


@tool
def get_stock_price(symbol: str) -> str:
    """Fetch latest stock price using Alpha Vantage."""
    try:
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            return "Alpha Vantage API key not found."
        
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        return str(data)
    except Exception as e:
        return f"Stock price error: {str(e)}"


tools = [tool_tavily_search, tool_wikipedia_search, tool_arxiv_search, get_stock_price]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    messages = state["messages"][-12:]   # Keep last 12 messages for context

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful, concise, and accurate AI assistant.
        - Always use conversation history when relevant.
        - Be clear and to the point.
        - STRICTLY Do not exceed ~500 tokens per response."""),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | llm_with_tools
    response = chain.invoke({"messages": messages})
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 5. Checkpointer
# -------------------
os.makedirs("Database", exist_ok=True)

def get_checkpointer():
    conn = sqlite3.connect("Database/chatbot.db", check_same_thread=False)
    return SqliteSaver(conn)

checkpointer = get_checkpointer()

# -------------------
# 6. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition, {"tools": "tools", END: END})
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 7. Helper
# -------------------
def retrieve_all_threads():
    """Return list of thread IDs"""
    try:
        threads = []
        for checkpoint in checkpointer.list(None):
            thread_id = checkpoint.config["configurable"]["thread_id"]
            threads.append(thread_id)
        return list(dict.fromkeys(threads))  # Remove duplicates while preserving order
    except:
        return []