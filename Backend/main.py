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

# FOR RAG
from typing import Any, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS    # Change to Chroma
from langchain_ollama import OllamaEmbeddings 


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

embeddings = OllamaEmbeddings(model="nomic-embed-text") 



# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


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


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [tool_tavily_search, tool_wikipedia_search, tool_arxiv_search, get_stock_price, rag_tool]
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
# 8. Helpers
# -------------------

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})

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