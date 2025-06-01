import os
from pathlib import Path
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


class Settings:
    """Application settings"""

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")

    LLM_MODEL = "mistral-saba-24b"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = "logs/chatbot.log"

    # Validation of name input
    MIN_NAME_LENGTH = 2

    @classmethod
    def validate(cls):
        """Validate required settings"""
        if not cls.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is required")
        
        # Ensure directory exists else create
        Path(cls.CHROMA_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(cls.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
        
