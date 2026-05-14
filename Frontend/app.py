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
            # with st.status("🔄 Indexing document...", expanded=True) as status:
            #     try:
            #         summary = ingest_pdf(uploaded_pdf.getvalue(), thread_id, uploaded_pdf.name)
            #         print("✅ Document Indexed Successfully")
            #         status.update(label="✅ Document Indexed Successfully", state="complete")
            #         st.success(f"**{summary['filename']}** loaded")
            #     except Exception as e:
            #         status.update(label="❌ Failed", state="error")
            #         st.error(str(e))
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

