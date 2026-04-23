# ==============================
# 1. LOAD ENV VARIABLES FIRST
# ==============================
from dotenv import load_dotenv
load_dotenv()

# ==============================
# 2. IMPORTS
# ==============================
import streamlit as st
import uuid
from datetime import datetime
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory


# ==============================
# 3. PAGE CONFIG
# ==============================``
st.set_page_config(
    page_title="Gemma Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==============================
# 4. SESSION STATE INIT
# ==============================
def new_chat_id():
    return str(uuid.uuid4())[:8]

if "chats" not in st.session_state:
    first_id = new_chat_id()
    st.session_state.chats = {
        first_id: {
            "title": "New conversation",
            "messages": [],
            "created_at": datetime.now().strftime("%H:%M"),
        }
    }
    st.session_state.active_chat = first_id

if "active_chat" not in st.session_state:
    st.session_state.active_chat = list(st.session_state.chats.keys())[0]

if "store" not in st.session_state:
    st.session_state.store = {}


# ==============================
# 5. MODEL & CHAIN (cached)
# ==============================
@st.cache_resource
def get_chain():
    model = ChatOllama(model="gemma:2b")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. ALWAYS use the conversation history to answer questions."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    return prompt | model

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

with_message_history = RunnableWithMessageHistory(get_chain(), get_session_history)


# ==============================
# 6. SIDEBAR (native Streamlit)
# ==============================
with st.sidebar:
    st.title("💬 Gemma Chat")
    st.caption("gemma:2b · local · Ollama")

    st.divider()

    if st.button("＋  New conversation", use_container_width=True, type="primary"):
        cid = new_chat_id()
        st.session_state.chats[cid] = {
            "title": "New conversation",
            "messages": [],
            "created_at": datetime.now().strftime("%H:%M"),
        }
        st.session_state.active_chat = cid
        st.rerun()

    st.divider()
    st.caption("CONVERSATIONS")

    for cid, chat in reversed(list(st.session_state.chats.items())):
        is_active = cid == st.session_state.active_chat
        title = chat["title"]
        label = f"▶ {title}" if is_active else title

        col1, col2 = st.columns([5, 1])
        with col1:
            if st.button(label, key=f"tab_{cid}", use_container_width=True):
                st.session_state.active_chat = cid
                st.rerun()
        with col2:
            if st.button("✕", key=f"del_{cid}"):
                if len(st.session_state.chats) > 1:
                    del st.session_state.chats[cid]
                    if cid in st.session_state.store:
                        del st.session_state.store[cid]
                    st.session_state.active_chat = list(st.session_state.chats.keys())[-1]
                    st.rerun()

    st.divider()
    st.caption("Ollama · LangChain · Streamlit")


# ==============================
# 7. MAIN CHAT AREA
# ==============================
active_id   = st.session_state.active_chat
active_chat = st.session_state.chats[active_id]

# Header
if active_chat['title'] == "New conversation":
    pass
else:
    st.subheader(f"💬 {active_chat['title']}")
    st.divider()

# Messages
messages = active_chat["messages"]

import streamlit.components.v1 as components

if not messages:
    components.html("""
    <style>
    body {
        margin: 0;
        font-family: sans-serif;
    }

    .empty-state-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 60px 20px;
        text-align: center;
    }

    h2 {
        font-size: 26px;
        font-weight: 500;
        margin-bottom: 10px;
    }

    p {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 40px;
    }

    .suggestion-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        max-width: 500px;
        width: 100%;
    }

    .suggestion-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px;
        text-align: left;
    }

    .icon { font-size: 18px; margin-bottom: 5px; }
    .title { font-size: 13px; font-weight: 500; }
    .sub { font-size: 12px; color: #6b7280; }

    .footer {
        margin-top: 30px;
        font-size: 11px;
        color: #9ca3af;
    }
    </style>

    <div class="empty-state-wrap">
        <h2>How can I help you?</h2>
        <p>Start typing below to begin a new conversation with Gemma running locally.</p>

        <div class="suggestion-grid">
            <div class="suggestion-card">
                <div class="icon">✍️</div>
                <div class="title">Help me write</div>
                <div class="sub">Drafts, emails, essays</div>
            </div>

            <div class="suggestion-card">
                <div class="icon">💡</div>
                <div class="title">Brainstorm ideas</div>
                <div class="sub">Projects, plans, concepts</div>
            </div>

            <div class="suggestion-card">
                <div class="icon">🔍</div>
                <div class="title">Explain something</div>
                <div class="sub">Concepts, how-tos, guides</div>
            </div>

            <div class="suggestion-card">
                <div class="icon">⚡</div>
                <div class="title">Write some code</div>
                <div class="sub">Scripts, functions, debug</div>
            </div>
        </div>

        <div class="footer">gemma:2b · running locally via Ollama</div>
    </div>
    """, height=400)



else:
    for msg in messages:
        role   = msg["role"]
        avatar = "🧑" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])


# ==============================
# 8. CHAT INPUT
# ==============================
if user_input := st.chat_input("Message Gemma…", key=f"input_{active_id}"):
    active_chat["messages"].append({"role": "user", "content": user_input})

    # Set conversation title from first message
    if len(active_chat["messages"]) == 1:
        active_chat["title"] = user_input[:40] + ("…" if len(user_input) > 40 else "")

    config = {"configurable": {"session_id": active_id}}
    with st.spinner("Thinking…"):
        response = with_message_history.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

    active_chat["messages"].append({"role": "assistant", "content": response.content})
    st.rerun()

