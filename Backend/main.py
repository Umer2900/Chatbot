from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Any, Dict, Optional
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
import tempfile

load_dotenv()

# ------------------- RAG Imports -------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings 


# -------------------
# 1. LLM + Embeddings
# -------------------
llm = ChatOllama( 
    model="gpt-oss:20b",  
    base_url="https://ollama.com", 
    client_kwargs={ 
        "headers": { 
            "Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}" 
        } 
    }, 
    temperature=0.3, 
)  

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# -------------------
# RAG Storage
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: str):
    return _THREAD_RETRIEVERS.get(str(thread_id))

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No file content received.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        thread_id_str = str(thread_id)
        _THREAD_RETRIEVERS[thread_id_str] = retriever
        _THREAD_METADATA[thread_id_str] = {
            "filename": filename or "Uploaded PDF",
            "documents": len(docs),
            "chunks": len(chunks),
        }
        return _THREAD_METADATA[thread_id_str]
    finally:
        try:
            os.remove(temp_path)
        except:
            pass


# -------------------
# 2. Tools
# -------------------
@tool
def rag_tool(query: str) -> str:
    """Retrieve relevant information from the PDF document uploaded in this chat thread."""
    thread_id = globals().get("thread_id")
    
    if not thread_id:
        return "Error: Could not determine current chat thread."

    retriever = _get_retriever(thread_id)
    if retriever is None:
        return "No document has been uploaded for this chat. Please upload a PDF first."

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content[:750] for doc in docs])
    filename = _THREAD_METADATA.get(str(thread_id), {}).get("filename", "PDF")

    return f"""Relevant information from the uploaded document '{filename}': {context}
            he user's question using the context above.""".strip()


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
llm_with_tools = llm.bind_tools(tools)


# -------------------
# 3. Updated State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str  


# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    thread_id = state.get("thread_id")
    # print(f"[DEBUG] chat_node called with thread_id: {thread_id}")   # Debugging

    # Make thread_id available to rag_tool 
    globals()["thread_id"] = thread_id   # Temporary but works

    messages = state["messages"][-12:]

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a helpful, concise, and accurate AI assistant.
            - **If you simulate tool usage, explicitly state: [Tool: tool_name] before the answer.**
            - Always use conversation history when relevant.
            - Be clear and to the point.
            - STRICTLY limit responses to ~400 tokens.

            **Equation Formatting Rules:**
            - Always write math/chemical equations in plain-text Unicode format.
            - Example:
                6 CO2 + 6 H2O + light energy -> C6H12O6 + 6 O2
                sin 3x + cos 3x = sqrt(2) sin 2x

            **Tool Usage Rules:**
            - Tools are used internally when required.
            - Do NOT expose internal tool call mechanics.
            - You may optionally mention: "I used a tool to compute/search this" AFTER giving the answer.
            
            **RAG Rules:**
            - If the user asks anything about the uploaded document, PDF, file, summary, or its content → **must use** the `rag_tool`.
            - Never answer from general knowledge when asked about the document.
         
            Current thread_id: {thread_id}
        """),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | llm_with_tools
    response = chain.invoke({"messages": messages})
    
    if response.tool_calls:
        print(f"[DEBUG] Tools called: {[t['name'] for t in response.tool_calls]}")
    
    return {"messages": [response], "thread_id": thread_id}


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
# Graph
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





# ------------------------------
# Thread Titles & Delete Support
# ------------------------------
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
