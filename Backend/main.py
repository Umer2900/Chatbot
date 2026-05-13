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
        from langchain_tavily import TavilySearch
        search = TavilySearch(max_results=3)
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
        return arxiv.run(query)[:2000]
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
            - **If you simulate tool usage, explicitly state: [Tool: tool_name] before the answer.**
            - Always use conversation history when relevant.
            - Be clear and to the point.
            - STRICTLY limit responses to ~400 tokens.

            Equation Formatting Rules:
            - Always write math/chemical equations in plain-text Unicode format.
            - Example:
                6 CO2 + 6 H2O + light energy -> C6H12O6 + 6 O2
                sin 3x + cos 3x = sqrt(2) sin 2x

            Tool Usage Rules:
            - Tools are used internally when required.
            - Do NOT expose internal tool call mechanics.
            - You may optionally mention: "I used a tool to compute/search this" AFTER giving the answer.
            
        """),
        
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
# Thread Titles & Delete Support
# -------------------
os.makedirs("Database", exist_ok=True)

def init_db():
    conn = sqlite3.connect("Database/chatbot.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS threads (
            thread_id TEXT PRIMARY KEY,
            title TEXT DEFAULT 'New Chat',
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def create_thread(thread_id: str, title: str = "New Chat"):
    conn = sqlite3.connect("Database/chatbot.db")
    now = datetime.now().isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO threads (thread_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (thread_id, title, now, now)
    )
    conn.commit()
    conn.close()

def update_thread_title(thread_id: str, title: str):
    conn = sqlite3.connect("Database/chatbot.db")
    conn.execute(
        "UPDATE threads SET title = ?, updated_at = ? WHERE thread_id = ?",
        (title, datetime.now().isoformat(), thread_id)
    )
    conn.commit()
    conn.close()

def delete_thread(thread_id: str):
    conn = sqlite3.connect("Database/chatbot.db")
    conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
    conn.commit()
    conn.close()
    
    try:
        checkpointer.delete({"configurable": {"thread_id": thread_id}})
    except:
        pass

def get_all_threads():
    """Return list of (thread_id, title)"""
    conn = sqlite3.connect("Database/chatbot.db")
    cursor = conn.execute("SELECT thread_id, title FROM threads ORDER BY updated_at DESC")
    threads = cursor.fetchall()
    conn.close()
    return threads