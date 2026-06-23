import sys
import os
import time
import streamlit as st
import requests
import uuid


# ====================== CONFIG ======================
BACKEND_URL = "http://127.0.0.1:8000"   # Change this for production

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="RAGent Chatbot • Enterprise Assistant",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    .main { background-color: #0E1117; color: #FAFAFA; }
    .stChatMessage { border-radius: 12px; padding: 14px 18px; }
    .user-message { background-color: #1E88E5 !important; color: white; }
    .assistant-message { background-color: #26334A !important; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
    .stMarkdown h1 { color: #4FC3F7; }
</style>
""", unsafe_allow_html=True)

# ====================== UTILITIES ======================
def generate_thread_id():
    return str(uuid.uuid4())

def get_all_threads():
    try:
        r = requests.get(f"{BACKEND_URL}/threads")
        return r.json() if r.status_code == 200 else []
    except:
        return []

def load_conversation(thread_id: str):
    """Load full conversation history from backend"""
    try:
        resp = requests.get(f"{BACKEND_URL}/messages/{thread_id}")
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"Failed to load conversation: {resp.status_code}")
            return []
    except Exception as e:
        print(f"Error loading conversation: {e}")
        return []
    
def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    st.session_state["title_generated"] = False
    requests.post(f"{BACKEND_URL}/threads", json={"thread_id": thread_id})
    st.session_state["chat_threads"] = get_all_threads()
    st.rerun()


# ====================== SESSION STATE ======================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["title_generated"] = False
    reset_chat()
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.markdown("# RAGent Chatbot")
    st.divider()

    if st.button("➕ New Conversation", use_container_width=True, type="primary"):
        reset_chat()

    st.divider()

    # Document RAG
    st.subheader("📄 Document Analysis")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")

    if uploaded_pdf:
        thread_id = st.session_state["thread_id"]
        with st.spinner("Indexing PDF..."):
            try:
                files = {"file": (uploaded_pdf.name, uploaded_pdf.getvalue(), "application/pdf")}
                r = requests.post(f"{BACKEND_URL}/upload-pdf/{thread_id}", files=files)
                if r.status_code == 200:
                    pass
                    # st.success(f"✅ {uploaded_pdf.name} indexed")
            except Exception as e:
                st.error(str(e))

    st.divider()

    # Conversations
    st.subheader("📂 Recent Conversations")
    for tid, title in st.session_state.get("chat_threads", []):
        col1, col2 = st.columns([4.5, 0.8])
        with col1:
            if st.button(title[:45] + "..." if len(title) > 45 else title, key=f"load_{tid}", use_container_width=True):
                st.session_state["thread_id"] = tid
                st.session_state["message_history"] = load_conversation(tid) 
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{tid}"):
                requests.delete(f"{BACKEND_URL}/threads/{tid}")
                st.session_state["chat_threads"] = get_all_threads()
                if st.session_state["thread_id"] == tid:
                    reset_chat()
                else:
                    st.rerun() 

# ====================== MAIN CHAT ======================
st.title("RAGent Chatbot")

current_title = next((t for tid, t in st.session_state.get("chat_threads", []) 
                     if tid == st.session_state["thread_id"]), "New Conversation")
title_bar = st.empty()
title_bar.subheader(f"📍 {current_title}")

# Display Chat History
for msg in st.session_state["message_history"]:
    with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])


# User Input
if user_input := st.chat_input("Ask anything..."):

    # Save user message
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    # Display user message
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(user_input)

    # Assistant response
    with st.chat_message("assistant", avatar="🤖"):

        placeholder = st.empty()

        try:

            # ================= NON-STREAMING REQUEST =================
            r = requests.post(
                f"{BACKEND_URL}/chat",
                json={
                    "message": user_input,
                    "thread_id": st.session_state["thread_id"]
                },
                timeout=90
            )

            # ================= SUCCESS =================
            if r.status_code == 200:

                data = r.json()

                full_response = data.get("response", "")

                # Final render (renders markdown tables correctly)
                placeholder.markdown(full_response)

            else:

                full_response = f"❌ Error: {r.text}"

                placeholder.markdown(full_response)

        except Exception as e:

            full_response = f"❌ Backend error: {str(e)}"

            placeholder.markdown(full_response)

    # Save assistant message
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": full_response
    })



    # Auto Title Generation
    if not st.session_state.get("title_generated", False) and len(st.session_state["message_history"]) >= 2:
        try:
            r = requests.post(
                f"{BACKEND_URL}/generate-title",
                json={
                    "message": user_input,
                    "thread_id": st.session_state["thread_id"]
                },
                stream=True,
                headers={"Accept": "text/event-stream"}
            )

            title_text = ""
            for line in r.iter_lines():
                if line:
                    text = line.decode("utf-8")
                    
                    # Remove SSE prefix
                    if text.startswith("data: "):
                        text = text.replace("data: ", "")

                    # Final title received
                    if "__FINAL__:" in text:
                        final_title = text.split("__FINAL__:")[-1].strip()
                        title_bar.subheader(f"📍 {final_title}")
                        st.session_state["title_generated"] = True
                        st.session_state["chat_threads"] = get_all_threads()
                        st.rerun()
                        break

                    # Streaming typing effect
                    else:
                        for char in text:
                            title_text += char
                            title_bar.subheader(f"📍 {title_text}▌")
                            time.sleep(0.015)   # typing speed

        except Exception as e:
            print("Title failed:", e)
            st.session_state["title_generated"] = True



# (Powershell)

# cd Frontend
# streamlit run app.py

#     OR

# streamlit run Frontend/app.py

#     OR 
# uv run python -m streamlit run Frontend/app.py








