import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

    # Chunking parameters
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    # Model parameters
    EMBEDDING_MODEL = "text-embedding-ada-002"
    CHAT_MODEL = "gpt-3.5-turbo"
    TEMPERATURE = 0.7

    # Vector store parameters
    INDEX_NAME = "financial-education-pdf"
    TOP_K = 5