from dotenv import load_dotenv
import os

load_dotenv()

# OLLAMA_MODEL = "gpt-oss:20b"
OLLAMA_MODEL = "gpt-oss:120b"
EMBEDDING_MODEL = "nomic-embed-text"

os.makedirs("Database", exist_ok=True)
DATABASE_PATH = "Database/chatbot.db"

