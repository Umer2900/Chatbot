
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from Backend.main import chatbot, get_all_threads, create_thread, update_thread_title, delete_thread
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    st.session_state["title_generated"] = False   # ← New flag
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

# ======================= Session State =======================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["title_generated"] = False
    create_thread(st.session_state["thread_id"], "New Chat")

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

# ============================ Sidebar ============================
st.sidebar.title("🧠 Chatbot")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)


st.sidebar.header("Conversations")

for thread_id, title in st.session_state["chat_threads"]:
    col1, col2 = st.sidebar.columns([4, 1])
    with col1:
        if st.button(title[:40] + "..." if len(title) > 40 else title, 
                    key=f"load_{thread_id}", use_container_width=True):
            st.session_state["thread_id"] = thread_id
            st.session_state["message_history"] = load_conversation(thread_id)
            st.rerun()
    
    with col2:
        if st.button("🗑", key=f"del_{thread_id}"):
            delete_thread(thread_id)
            st.session_state["chat_threads"] = get_all_threads()
            if st.session_state["thread_id"] == thread_id:
                reset_chat()
            else:
                st.rerun()

# ============================ Main Chat ============================
st.title("LangGraph Chatbot")

current_title = next((t for tid, t in st.session_state["chat_threads"] if tid == st.session_state["thread_id"]), "New Chat")
st.subheader(current_title)

# Display chat history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_input := st.chat_input("Type your message..."):
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

        for event in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="values"
        ):
            if "messages" in event:
                last_message = event["messages"][-1]
                if isinstance(last_message, AIMessage) and last_message.content:
                    full_response = last_message.content
                    message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state["message_history"].append({"role": "assistant", "content": full_response})

    # ==================== AUTO TITLE GENERATION (Only Once) ====================
    if not st.session_state.get("title_generated", False) and len(st.session_state["message_history"]) >= 2:
        try:
            from Backend.main import llm
            from langchain_core.messages import HumanMessage

            title_prompt = f"""Generate a very short title (3 words max) for this conversation.
                            Return ONLY the title. No quotes, no explanation.

                            User: {user_input}
                            Assistant: {full_response[:180]}"""

            new_title = llm.invoke([HumanMessage(content=title_prompt)]).content
            new_title = new_title.strip().strip('"').strip("'").strip()
            
            if len(new_title.split()) > 6:
                new_title = " ".join(new_title.split()[:6])
            
            if 3 < len(new_title) < 45:
                update_thread_title(st.session_state["thread_id"], new_title)
                st.session_state["chat_threads"] = get_all_threads()
                st.session_state["title_generated"] = True
                st.rerun()                    # Immediate refresh
        except Exception as e:
            print("Title generation failed:", str(e))
            st.session_state["title_generated"] = True  # Prevent repeated attempts