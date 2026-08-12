# 📚 DocsGPT

DocsGPT is a Retrieval-Augmented Generation (RAG) application that answers questions using a collection of Markdown documents.

The application reads Markdown files, splits them into chunks, generates embeddings using Gemini, stores the vectors in Pinecone, retrieves relevant chunks for user questions, and generates grounded answers using Groq.

The application is deployed using Streamlit.

## Architecture

```text
Markdown Documents
       │
       ▼
   chunker.py
       │
       ▼
   Text Chunks
       │
       ▼
  embedder.py
       │
       ▼
Gemini Embeddings
   1536 dimensions
       │
       ▼
    store.py
       │
       ▼
    Pinecone
       │
       │
       │ User Question
       ▼
   Gemini Query
    Embedding
       │
       ▼
    Pinecone
    Retrieval
       │
       ▼
 Relevant Chunks
       │
       ▼
      Groq
       │
       ▼
 Answer + Citations
       │
       ▼
   Streamlit UI