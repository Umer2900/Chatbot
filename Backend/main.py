"""
Backend/
├── main.py                 # FastAPI App (Entry Point)
├── config.py               # Configuration & Settings
├── database.py             # Database & Thread Management
├── llm.py                  # LLM Used
├── llm_graph.py            # Graph, Chat Logic
├── rag.py                  # RAG Logic (PDF Ingestion + Retriever)
├── tools.py                # All Tools
└── models.py               # Pydantic Models    
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
import uuid
import traceback

# Import from sibling modules (absolute import)
from .models import ChatRequest
from .database import init_db, create_thread, get_all_threads, delete_thread, get_async_checkpointer, update_thread_title
from .rag import ingest_pdf
from .llm_graph import graph   
from .llm import llm

from contextlib import asynccontextmanager
chatbot = None    # Global chatbot variable

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chatbot
    checkpointer = await get_async_checkpointer()
    chatbot = graph.compile(checkpointer=checkpointer)
    print("✅ Chatbot initialized with AsyncSqliteSaver")
    try:
        yield
    finally:
        print("Shutting down...")

app = FastAPI(title="Chatbot API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

# ====================== ENDPOINTS ======================

@app.post("/chat")
async def chat(request: ChatRequest):

    thread_id = request.thread_id or str(uuid.uuid4())

    input_state = {
        "messages": [HumanMessage(content=request.message)],
        "thread_id": thread_id
    }

    try:

        result = await chatbot.ainvoke(
            input_state,
            config={"configurable": {"thread_id": thread_id}},
        )

        # Get final assistant message
        messages = result.get("messages", [])

        final_response = ""

        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                final_response = msg.content
                break

        return {
            "response": final_response,
            "thread_id": thread_id
        }

    except Exception as e:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/generate-title")
async def generate_title(request: ChatRequest):

    async def title_stream():

        try:

            title_prompt = f"""
                        Generate a very short, meaningful title (maximum 3 words).
                        Return ONLY the title.

                        User: {request.message}
                        """

            title = ""

            for chunk in llm.stream(
                [HumanMessage(content=title_prompt)]
            ):

                if chunk.content:

                    title += chunk.content

                    yield f"data: {chunk.content}\n\n"

            final_title = title.strip().strip('"').strip("'")[:50]

            update_thread_title(
                request.thread_id,
                final_title
            )

            yield f"data: __FINAL__:{final_title}\n\n"

        except Exception:
            yield "data: __FINAL__:New Conversation\n\n"

    return StreamingResponse(
        title_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/upload-pdf/{thread_id}")
async def upload_pdf(thread_id: str, file: UploadFile = File(...)):
    try:
        content = await file.read()
        summary = ingest_pdf(content, thread_id, file.filename)
        return {"status": "success", "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/threads")
async def list_threads():
    return get_all_threads()

@app.post("/threads")
async def new_thread(request: dict):
    thread_id = request.get("thread_id", str(uuid.uuid4()))
    create_thread(thread_id)
    return {"thread_id": thread_id, "title": "New Chat"}

@app.delete("/threads/{thread_id}")
async def delete_thread_api(thread_id: str):
    delete_thread(thread_id)
    return {"status": "success"}

@app.get("/messages/{thread_id}")
async def get_messages(thread_id: str):
    """Get full conversation history for a thread"""
    try:
        state = await chatbot.aget_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        messages = []
        for msg in state.values.get("messages", []):
            if isinstance(msg, HumanMessage):
                messages.append({
                    "role": "user",
                    "content": msg.content
                })
            elif isinstance(msg, AIMessage) and msg.content:
                messages.append({
                    "role": "assistant",
                    "content": msg.content
                })
        return messages

    except Exception as e:
        print(f"Error loading messages for {thread_id}: {e}")
        return []



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Backend.main:app", host="0.0.0.0", port=8000, reload=True)
    




"""

(Powershell)

cd Backend
pip install fastapi uvicorn python-multipart
uvicorn main:app --reload --port 8000

OR

uvicorn Backend.main:app --reload --port 8000

OR

uvicorn Backend.main:app --port 8000

OR

uv run python -m uvicorn Backend.main:app --reload --reload-dir Backend --port 8000


"""
