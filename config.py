import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from google import genai
from pinecone import Pinecone


# Load environment variables
load_dotenv()


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"


# --------------------------------------------------
# API Keys
# --------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


def validate_environment():
    """Check that all required API keys are available."""

    missing = []

    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")

    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")

    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")

    if missing:
        raise RuntimeError(
            f"Missing environment variables: {', '.join(missing)}"
        )


validate_environment()


# --------------------------------------------------
# Models
# --------------------------------------------------

GROQ_MODEL = "llama-3.3-70b-versatile"

GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

EMBEDDING_DIMENSION = 1536


# --------------------------------------------------
# Chunking configuration
# --------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


# --------------------------------------------------
# Retrieval configuration
# --------------------------------------------------

TOP_K = 5

# Tune this after testing your own documents.
# Pinecone cosine similarity normally returns values
# where larger means more similar.
MIN_SIMILARITY = 0.45


# --------------------------------------------------
# Pinecone configuration
# --------------------------------------------------

PINECONE_INDEX_NAME = "docs-gpt"

PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"


# --------------------------------------------------
# Prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are DocsGPT, a document question-answering assistant.

Your job is to answer questions ONLY using the supplied document context.

Rules:

1. Use only the provided context.
2. Never use outside knowledge to answer.
3. If the context does not contain enough information, say exactly:
   "I don't have that in the docs."
4. Every factual statement must have a citation such as [1], [2], etc.
5. The citation number must correspond to the numbered context source.
6. Do not invent citations.
7. Do not mention information that cannot be supported by the context.
8. Keep answers clear and concise.
"""


# --------------------------------------------------
# Clients
# --------------------------------------------------

groq_client = Groq(api_key=GROQ_API_KEY)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)