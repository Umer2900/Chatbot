import sqlite3
import aiosqlite
from datetime import datetime
from .config import DATABASE_PATH 
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def get_async_checkpointer():
    """Return AsyncSqliteSaver with proper aiosqlite connection"""
    conn = await aiosqlite.connect(DATABASE_PATH)
    return AsyncSqliteSaver(conn)

# def get_checkpointer():   # For backward compatibility
#     conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
#     from langgraph.checkpoint.sqlite import SqliteSaver
#     return SqliteSaver(conn)

def init_db():
    """Sync initialization"""
    conn = sqlite3.connect(DATABASE_PATH)
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

def create_thread(thread_id: str, title: str = "New Chat"):
    conn = sqlite3.connect(DATABASE_PATH)
    now = datetime.now().isoformat()
    conn.execute(
        "INSERT OR REPLACE INTO threads (thread_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (thread_id, title, now, now)
    )
    conn.commit()
    conn.close()

def update_thread_title(thread_id: str, title: str):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute(
        "UPDATE threads SET title = ?, updated_at = ? WHERE thread_id = ?",
        (title, datetime.now().isoformat(), thread_id)
    )
    conn.commit()
    conn.close()

def delete_thread(thread_id: str):
    conn = sqlite3.connect(DATABASE_PATH)
    conn.execute("DELETE FROM threads WHERE thread_id = ?", (thread_id,))
    conn.commit()
    conn.close()

def get_all_threads():
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.execute("SELECT thread_id, title FROM threads ORDER BY updated_at DESC")
    threads = cursor.fetchall()
    conn.close()
    return threads