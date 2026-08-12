from chunker import create_chunks
from embedder import embed_documents
from store import upsert_chunks, get_vector_count


def main():

    print("=" * 60)
    print("DocsGPT Ingestion")
    print("=" * 60)

    # 1. Read and chunk documents
    print("\n[1/3] Reading and chunking documents...")

    chunks = create_chunks()

    if not chunks:
        raise RuntimeError(
            "No markdown files found in the docs/ folder."
        )

    print(f"Created {len(chunks)} chunks.")

    # 2. Generate embeddings
    print("\n[2/3] Generating Gemini embeddings...")

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embed_documents(texts)

    print(f"Generated {len(embeddings)} embeddings.")

    # 3. Store in Pinecone
    print("\n[3/3] Upserting vectors into Pinecone...")

    upserted = upsert_chunks(
        chunks,
        embeddings,
    )

    print(f"Upserted {upserted} chunks.")

    # Show final count
    count = get_vector_count()

    print("\n" + "=" * 60)
    print(f"Total vectors in Pinecone: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()