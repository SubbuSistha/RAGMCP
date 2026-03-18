"""
embedding.py
-----------------
Step 2 of RAG Pipeline — Embed and Store

What this file does:
  - Loads chunks from 01_chunk.py
  - Converts each chunk content into a vector (embedding)
  - Stores vectors + content + metadata into ChromaDB
  - Prints progress so you can see what is happening

Run:
  uv run python src/basic/embedding.py
"""

import json
import os

import chromadb
from sentence_transformers import SentenceTransformer


# ── Config ────────────────────────────────────────────────────────────────────

CHUNKS_PATH      = "embedding/chunks.json"
CHROMA_DB_PATH   = "chroma_db"
COLLECTION_NAME  = "hr_policy"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"


# ── Step 1: Load chunks from JSON ─────────────────────────────────────────────

def load_chunks(path: str) -> list[dict]:
    """Load chunks created by 01_chunk.py"""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Chunks file not found: {path}\n"
            f"Run 01_chunk.py first."
        )

    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"✓ Loaded {len(chunks)} chunks from {path}\n")
    return chunks


# ── Step 2: Load embedding model ──────────────────────────────────────────────

def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Load the sentence transformer model.

    First run   → downloads ~80MB model to local cache
    Next runs   → loads from cache instantly

    Model: all-MiniLM-L6-v2
      - Fast and lightweight
      - Produces 384-dimensional vectors
      - Good quality for English text
      - Perfect for training demos
    """

    print(f"Loading embedding model: {model_name}")
    print("(First run downloads ~80MB — subsequent runs load from cache)\n")

    model = SentenceTransformer(model_name)

    print(f"✓ Model loaded")
    print(f"✓ Vector dimensions: {model.get_sentence_embedding_dimension()}\n")

    return model


# ── Step 3: Setup ChromaDB ────────────────────────────────────────────────────

def setup_chromadb(db_path: str, collection_name: str) -> chromadb.Collection:
    """
    Create a persistent ChromaDB client and collection.

    persistent client → saves to disk at chroma_db/
    collection        → like a table in a normal database
                        stores vectors + content + metadata together
    """

    print(f"Setting up ChromaDB at: {db_path}")

    # PersistentClient saves data to disk
    # Data survives between runs — no need to re-embed every time
    client = chromadb.PersistentClient(path=db_path)

    # Delete existing collection if it exists
    # This ensures fresh start every time you run this script
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"  ↻ Deleted existing collection: {collection_name}")

    # Create fresh collection
    collection = client.get_or_create_collection(
        name     = collection_name,
        metadata = {"hnsw:space": "cosine"},  # use cosine similarity
    )

    print(f"✓ Collection ready: {collection_name}\n")
    return collection


# ── Step 4: Embed and store chunks ────────────────────────────────────────────

def embed_and_store(
    chunks     : list[dict],
    model      : SentenceTransformer,
    collection : chromadb.Collection,
) -> None:
    """
    Embed each chunk and store in ChromaDB.

    For each chunk:
      1. Take the content text
      2. Pass through embedding model → get vector (384 numbers)
      3. Store in ChromaDB with:
           id       → unique identifier
           embedding→ the 384 number vector
           document → original text content
           metadata → section, parent, doc_name etc.
    """

    print("Embedding and storing chunks...")
    print("-" * 60)

    ids         = []
    embeddings  = []
    documents   = []
    metadatas   = []

    for chunk in chunks:
        content  = chunk["content"]
        metadata = chunk["metadata"]
        idx      = metadata["chunk_index"]

        # Embed the content
        vector = model.encode(content).tolist()

        # Show what is happening for each chunk
        print(f"  Chunk {idx:02d} | {metadata['section'][:45]:<45} | {len(vector)} dims")

        # Collect for batch insert
        ids.append(f"chunk_{idx:03d}")
        embeddings.append(vector)
        documents.append(content)
        metadatas.append({
            "doc_name"      : metadata["doc_name"],
            "section"       : metadata["section"],
            "parent_section": metadata["parent_section"],
            "chunk_index"   : metadata["chunk_index"],
            "word_count"    : metadata["word_count"],
        })

    # Store everything in ChromaDB in one batch call
    collection.add(
        ids        = ids,
        embeddings = embeddings,
        documents  = documents,
        metadatas  = metadatas,
    )

    print("-" * 60)
    print(f"\n✓ Stored {len(ids)} chunks in ChromaDB")


# ── Step 5: Verify storage ────────────────────────────────────────────────────

def verify_storage(collection: chromadb.Collection) -> None:
    """
    Quick verification — show what is stored in ChromaDB.
    Peek at first 3 chunks to confirm everything looks right.
    """

    total = collection.count()
    print(f"\n✓ ChromaDB collection count: {total} chunks")

    print("\nPeeking at first 3 stored chunks:")
    print("-" * 60)

    result = collection.peek(limit=3)

    for i in range(len(result["ids"])):
        print(f"\n  ID       : {result['ids'][i]}")
        print(f"  Section  : {result['metadatas'][i]['section']}")
        print(f"  Parent   : {result['metadatas'][i]['parent_section']}")
        print(f"  Words    : {result['metadatas'][i]['word_count']}")
        print(f"  Vector   : [{result['embeddings'][i][0]:.4f}, "
              f"{result['embeddings'][i][1]:.4f}, "
              f"{result['embeddings'][i][2]:.4f} ... "
              f"{len(result['embeddings'][i])} dims total]")
        print(f"  Preview  : {result['documents'][i][:100].replace(chr(10), ' ')}...")

    print("-" * 60)


# ── Step 6: Demo — show embedding live ───────────────────────────────────────

def demo_embedding(model: SentenceTransformer) -> None:
    """
    Training demo — show a sentence becoming a vector live.
    This is the most important concept to show your audience.
    """

    print("\n" + "=" * 60)
    print("DEMO — Watch a sentence become a vector")
    print("=" * 60)

    sentences = [
        "Employees get 12 days of sick leave per year",
        "Staff are entitled to twelve sick days annually",
        "How many sick leaves do I get?",
        "What is the work from home policy?",         # different topic
    ]

    print("\nSame meaning → similar vectors (close cosine distance)")
    print("Different meaning → different vectors (far cosine distance)\n")

    vectors = model.encode(sentences)

    for i, (sentence, vector) in enumerate(zip(sentences, vectors)):
        print(f"  [{i+1}] \"{sentence}\"")
        print(f"       Vector (first 5 of 384): "
              f"[{vector[0]:.3f}, {vector[1]:.3f}, {vector[2]:.3f}, "
              f"{vector[3]:.3f}, {vector[4]:.3f}]")
        print()

    # Show cosine similarity between sentences
    from sentence_transformers import util

    print("Cosine Similarity (1.0 = identical, 0.0 = unrelated):")
    print(f"  Sentence 1 vs 2 (same meaning)   : "
          f"{util.cos_sim(vectors[0], vectors[1]).item():.4f}  ← high")
    print(f"  Sentence 1 vs 3 (question vs doc): "
          f"{util.cos_sim(vectors[0], vectors[2]).item():.4f}  ← medium")
    print(f"  Sentence 1 vs 4 (diff topic)     : "
          f"{util.cos_sim(vectors[0], vectors[3]).item():.4f}  ← low")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load chunks from previous step
    chunks = load_chunks(CHUNKS_PATH)

    # 2. Load embedding model
    model = load_embedding_model(EMBEDDING_MODEL)

    # 3. Setup ChromaDB
    collection = setup_chromadb(CHROMA_DB_PATH, COLLECTION_NAME)

    # 4. Embed and store all chunks
    embed_and_store(chunks, model, collection)

    # 5. Verify what is stored
    verify_storage(collection)

    # 6. Demo — show embedding live (great for training)
    demo_embedding(model)

    print("\n✓ Embedding complete. ChromaDB is ready.")


if __name__ == "__main__":
    main()