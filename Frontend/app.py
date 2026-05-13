import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from Backend.main import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid

# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].insert(0, thread_id)

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

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

# Add current thread if not present
if st.session_state["thread_id"] not in st.session_state["chat_threads"]:
    st.session_state["chat_threads"].insert(0, st.session_state["thread_id"])

# ============================ Sidebar ============================
st.sidebar.title("🧠 Chatbot")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.header("Conversations")
for thread_id in st.session_state["chat_threads"][:15]:  # Limit displayed threads
    if st.sidebar.button(str(thread_id)[:8] + "...", key=thread_id):
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = load_conversation(thread_id)
        st.rerun()

# ============================ Main Chat ============================
st.title("LangGraph Chatbot")

# Display chat history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if user_input := st.chat_input("Type your message..."):
    # Add user message
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant response
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

    # Save assistant response
    st.session_state["message_history"].append({"role": "assistant", "content": full_response})

    # Optional: Auto refresh thread list
    if st.session_state["thread_id"] not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].insert(0, st.session_state["thread_id"])