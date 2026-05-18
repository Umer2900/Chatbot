# Backend/main.py         __init__.py
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Any, Dict, Optional
from langchain_core.messages import BaseMessage
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
# RAG Storage (Per Thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: str):
    return _THREAD_RETRIEVERS.get(str(thread_id))

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """Ingest PDF and create retriever for the specific thread."""
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
def rag_tool(query: str, thread_id: Optional[str] = None) -> str:
    """Retrieve relevant information from the PDF document uploaded in this chat thread."""
    if not thread_id:
        return "Error: Could not determine current chat thread."

    retriever = _get_retriever(thread_id)
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
llm_with_tools = llm.bind_tools(tools)


# -------------------
# 3. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str


# -------------------
# 4. Nodes
# -------------------
def chat_node(state: ChatState):
    messages = state["messages"][-12:]
    thread_id = state.get("thread_id")

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
# 7. Helpers
# -------------------
def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


# ------------------------------
# Thread Titles & Delete Support
# ------------------------------
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



"""
pip install fastapi uvicorn python-multipart

cd Backend
python main.py    OR       uvicorn Backend.main:app --reload --port 8000


"""



# Frontend/app.py
import sys
import os
import time
import streamlit as st
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Backend.main import (
    chatbot, get_all_threads, create_thread, update_thread_title,
    delete_thread, ingest_pdf, thread_has_document, thread_document_metadata
)
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="CorpAI • Enterprise Assistant",
    page_icon="💼",
    layout="centered",
    # layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CORPORATE CSS ======================
st.markdown("""
<style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 14px 18px;
    }
    .user-message {
        background-color: #1E88E5 !important;
        color: white;
    }
    .assistant-message {
        background-color: #26334A !important;
    }
    .sidebar .stButton button {
        border-radius: 8px;
        padding: 8px 16px;
    }
    .chat-container {
        background-color: #161B26;
        border-radius: 12px;
        padding: 20px;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
    }
    .stMarkdown h1 {
        color: #4FC3F7;
    }
</style>
""", unsafe_allow_html=True)


# ====================== UTILITIES ======================
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    st.session_state["title_generated"] = False
    create_thread(thread_id, "New Chat")
    st.session_state["chat_threads"] = get_all_threads()
    st.rerun()

def load_conversation(thread_id):
    try:
        state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
        messages = state.values.get("messages", [])
        display = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                display.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) and msg.content:
                display.append({"role": "assistant", "content": msg.content})
        return display
    except:
        return []


# ====================== SESSION STATE ======================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["title_generated"] = False
    create_thread(st.session_state["thread_id"], "New Chat")
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()
if "render_counter" not in st.session_state:
    st.session_state["render_counter"] = 0

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("# 💼 Multi Utility Chatbot")
    st.divider()

    if st.button("➕ New Conversation", use_container_width=True, type="primary"):
        reset_chat()

    st.divider()

    # Document RAG Section
    st.subheader("📄 Document Analysis")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader", label_visibility="collapsed")

    if uploaded_pdf is not None:
        thread_id = st.session_state["thread_id"]
        if not thread_has_document(thread_id) or uploaded_pdf.name not in str(thread_document_metadata(thread_id)):
            try:
                summary = ingest_pdf(uploaded_pdf.getvalue(), thread_id, uploaded_pdf.name)
                print("✅ Document Indexed Successfully")
            except Exception as e:
                st.error(str(e))


    st.divider()

    # Conversations
    st.subheader("📂 Recent Conversations")
    threads_placeholder = st.empty()

    def render_threads(animated_title=None):
        with threads_placeholder.container():
            for tid, title in st.session_state["chat_threads"]:
                col1, col2 = st.columns([4.5, 0.8])
                with col1:
                    display_title = animated_title if (animated_title and tid == st.session_state["thread_id"]) else title
                    if st.button(
                        display_title[:45] + "..." if len(display_title) > 45 else display_title,
                        key=f"load_{tid}_{st.session_state['render_counter']}",
                        use_container_width=True
                    ):
                        st.session_state["thread_id"] = tid
                        st.session_state["message_history"] = load_conversation(tid)
                        st.rerun()
                with col2:
                    if st.button("🗑", key=f"del_{tid}_{st.session_state['render_counter']}"):
                        delete_thread(tid)
                        st.session_state["chat_threads"] = get_all_threads()
                        if st.session_state["thread_id"] == tid:
                            reset_chat()
                        else:
                            st.rerun()

    render_threads()

# ====================== MAIN DASHBOARD ======================

st.title("Multi Utility Chatbot")

# Current Chat Title
current_title = next((t for tid, t in st.session_state["chat_threads"] if tid == st.session_state["thread_id"]),"New Conversation")

title_bar = st.empty()
title_bar.subheader(f"📍 {current_title}")

# Chat Container
chat_container = st.container()
with chat_container:
    for message in st.session_state["message_history"]:
        with st.chat_message(message["role"], avatar="🧑‍💼" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

# User Input
if user_input := st.chat_input("Ask anything..."):
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        full_response = ""
        previous_text = ""

        CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}
        input_state = {
            "messages": [HumanMessage(content=user_input)],
            "thread_id": st.session_state["thread_id"]
        }

        for event in chatbot.stream(input_state, config=CONFIG, stream_mode="values"):
            if "messages" in event:
                last_message = event["messages"][-1]
                if isinstance(last_message, AIMessage) and last_message.content:
                    new_text = last_message.content
                    diff = new_text[len(previous_text):]
                    previous_text = new_text

                    for ch in diff:
                        full_response += ch
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)

        message_placeholder.markdown(full_response)

    st.session_state["message_history"].append({"role": "assistant", "content": full_response})


    # ==================== AUTO TITLE GENERATION ====================

    if not st.session_state.get("title_generated", False) and len(st.session_state["message_history"]) >= 2:
        try:
            from Backend.main import llm

            title_prompt = f"""Generate a very short title (3 words max) for this conversation.
                            Return ONLY the title. No quotes, no explanation.

                            User: {user_input}
                            Assistant: {full_response[:180]}"""

            streamed_title = ""

            for chunk in llm.stream([HumanMessage(content=title_prompt)]):
                if chunk.content:
                    for ch in chunk.content:
                        streamed_title += ch

                        # update top title
                        title_bar.subheader(streamed_title + "▌")

                        # IMPORTANT: update counter each render (unique keys)
                        st.session_state["render_counter"] += 1
                        render_threads(animated_title=streamed_title + "▌")

                        time.sleep(0.03)

            streamed_title = streamed_title.strip().strip('"').strip("'").strip()

            if len(streamed_title.split()) > 6:
                streamed_title = " ".join(streamed_title.split()[:6])

            title_bar.subheader(streamed_title)

            update_thread_title(st.session_state["thread_id"], streamed_title)
            st.session_state["chat_threads"] = get_all_threads()
            st.session_state["title_generated"] = True

            time.sleep(0.4)
            st.rerun()

        except Exception as e:
            print("Title generation failed:", str(e))
            st.session_state["title_generated"] = True

