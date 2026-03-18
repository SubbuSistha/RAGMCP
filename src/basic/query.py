"""
03_query.py
-----------
Step 3 of RAG Pipeline — Query, Retrieve and Answer

What this file does:
  - Takes a user question
  - Embeds the question into a vector
  - Searches ChromaDB for closest matching chunks
  - Sends question + retrieved chunks to Google Gemini
  - Returns a grounded answer with source reference

Run:
  uv run python src/basic/query.py
"""

import os

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DB_PATH  = "chroma_db"
COLLECTION_NAME = "hr_policy"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K           = 3
GEMINI_MODEL    = "gemini-3-flash-preview"   # fast and free tier available

load_dotenv()


# ── Step 1: Load embedding model ──────────────────────────────────────────────

def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load the same model used during embed_store step."""
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"✓ Model loaded\n")
    return model


# ── Step 2: Connect to ChromaDB ───────────────────────────────────────────────

def load_collection(db_path: str, collection_name: str) -> chromadb.Collection:
    """
    Connect to existing ChromaDB and load collection.
    Reads from disk — no re-embedding needed.
    """

    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"ChromaDB not found at: {db_path}\n"
            f"Run 02_embed_store.py first."
        )

    client     = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name=collection_name)

    print(f"✓ Connected to ChromaDB")
    print(f"✓ Collection: {collection_name} ({collection.count()} chunks)\n")

    return collection


# ── Step 3: Setup Google Gemini ───────────────────────────────────────────────

def setup_gemini() -> genai.GenerativeModel:
    """
    Configure Google Gemini API.

    Model: gemini-1.5-flash
      - Fast response time
      - Free tier available
      - Good for RAG — follows instructions well
      - Large context window — handles multiple chunks easily
    """

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found.\n"
            "Add it to your .env file: GOOGLE_API_KEY=your_key_here"
        )

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name = GEMINI_MODEL,
        generation_config = genai.GenerationConfig(
            temperature       = 0,      # deterministic answers for RAG
            max_output_tokens = 1024,
        )
    )

    print(f"✓ Gemini model ready: {GEMINI_MODEL}\n")
    return model


# ── Step 4: Embed the question ────────────────────────────────────────────────

def embed_question(question: str, model: SentenceTransformer) -> list[float]:
    """
    Convert user question into a vector.
    Same process as embedding document chunks.
    This vector is used to search ChromaDB.
    """

    vector = model.encode(question).tolist()
    print(f"✓ Question embedded → {len(vector)} dimensional vector\n")
    return vector


# ── Step 5: Search ChromaDB ───────────────────────────────────────────────────

def search_chunks(
    question_vector : list[float],
    collection      : chromadb.Collection,
    top_k           : int,
) -> list[dict]:
    """
    Search ChromaDB for chunks closest to the question vector.

    ChromaDB returns:
      ids        → chunk identifiers
      documents  → original text content
      metadatas  → section, parent, doc_name etc.
      distances  → cosine distance (lower = more similar)
    """

    results = collection.query(
        query_embeddings = [question_vector],
        n_results        = top_k,
        include          = ["documents", "metadatas", "distances"],
    )

    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "id"      : results["ids"][0][i],
            "content" : results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
            "score"   : round(1 - results["distances"][0][i], 4),
        })

    return retrieved


# ── Step 6: Print retrieved chunks ───────────────────────────────────────────

def print_retrieved_chunks(chunks: list[dict]) -> None:
    """
    Show which chunks were retrieved and their similarity scores.
    Important for training — audience sees what RAG found.
    """

    print("=" * 60)
    print(f"RETRIEVED CHUNKS (Top {len(chunks)})")
    print("=" * 60)

    for i, chunk in enumerate(chunks):
        print(f"\n[{i+1}] {chunk['metadata']['section']}")
        print(f"     Parent  : {chunk['metadata']['parent_section']}")
        print(f"     Score   : {chunk['score']} (1.0 = perfect match)")
        print(f"     Preview : {chunk['content'][:150].replace(chr(10), ' ')}...")

    print("=" * 60)


# ── Step 7: Build prompt for Gemini ──────────────────────────────────────────

def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Build the prompt that goes to Gemini.

    Structure:
      - Role: HR assistant context
      - Retrieved chunks: actual content from documents
      - Question: what the user asked
      - Instruction: answer only from provided context
    """

    context_blocks = []
    for i, chunk in enumerate(chunks):
        block = (
            f"[Source {i+1}: {chunk['metadata']['section']}]\n"
            f"{chunk['content']}"
        )
        context_blocks.append(block)

    context = "\n\n".join(context_blocks)

    prompt = f"""You are an HR assistant for TechNova Solutions.
Answer the employee's question using ONLY the context provided below.
If the answer is not in the context, say "I could not find this information in the HR policy document."
Always mention which section your answer is from.

---CONTEXT START---
{context}
---CONTEXT END---

Employee Question: {question}

Answer:"""

    return prompt


# ── Step 8: Ask Gemini ────────────────────────────────────────────────────────

def ask_gemini(prompt: str, gemini_model: genai.GenerativeModel) -> str:
    """
    Send the prompt to Gemini and get the answer.

    Gemini receives:
      - The retrieved chunks as context
      - The user question
      - Instruction to answer only from context

    This is the Augmented Generation part of RAG.
    """

    response = gemini_model.generate_content(prompt)
    return response.text


# ── Step 9: Full RAG query pipeline ──────────────────────────────────────────

def rag_query(
    question     : str,
    embed_model  : SentenceTransformer,
    gemini_model : genai.GenerativeModel,
    collection   : chromadb.Collection,
    top_k        : int = TOP_K,
    verbose      : bool = True,
) -> tuple[str, list[dict]]:
    """
    Full RAG pipeline for one question.

    Steps:
      1. Embed the question
      2. Search ChromaDB → get top K chunks
      3. Build prompt with context
      4. Ask Gemini → get answer

    Returns answer string and retrieved chunks.
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}\n")

    # 1. Embed question
    question_vector = embed_question(question, embed_model)

    # 2. Search ChromaDB
    retrieved_chunks = search_chunks(question_vector, collection, top_k)

    # 3. Show retrieved chunks
    if verbose:
        print_retrieved_chunks(retrieved_chunks)

    # 4. Build prompt
    prompt = build_prompt(question, retrieved_chunks)

    if verbose:
        print("\nSending to Gemini...")

    # 5. Ask Gemini
    answer = ask_gemini(prompt, gemini_model)

    return answer, retrieved_chunks


# ── Demo questions ────────────────────────────────────────────────────────────

def run_demo(
    embed_model  : SentenceTransformer,
    gemini_model : genai.GenerativeModel,
    collection   : chromadb.Collection,
) -> None:
    """
    Run preset demo questions to show RAG in action.
    Mix of good RAG questions and one bad RAG question.
    Perfect for live training demonstration.
    """

    questions = [
        # Good RAG questions — semantic search works well
        "How many sick leaves do I get per year?",
        "What is the work from home policy for managers?",
        "What happens when I resign from the company?",
        "Can I carry forward my earned leave?",

        # Bad RAG question — show RAG limitation honestly
        "How many total employees work at TechNova?",
    ]

    for question in questions:
        answer, chunks = rag_query(
            question, embed_model, gemini_model, collection, verbose=True
        )

        print(f"\n{'='*60}")
        print("FINAL ANSWER FROM GEMINI:")
        print(f"{'='*60}")
        print(answer)
        print(f"\n  Sources used:")
        for chunk in chunks:
            print(f"    → {chunk['metadata']['section']} "
                  f"(score: {chunk['score']})")
        print(f"\n{'*'*60}\n")

        input("Press Enter for next question...")


# ── Interactive mode ──────────────────────────────────────────────────────────

def run_interactive(
    embed_model  : SentenceTransformer,
    gemini_model : genai.GenerativeModel,
    collection   : chromadb.Collection,
) -> None:
    """
    Interactive mode — type your own questions.
    Type 'exit' to quit.
    """

    print("\n" + "=" * 60)
    print("INTERACTIVE MODE — Ask anything about TechNova HR Policy")
    print("Type 'exit' to quit")
    print("=" * 60)

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in ("exit", "quit", "q"):
            print("Exiting. Goodbye!")
            break

        if not question:
            continue

        answer, chunks = rag_query(
            question, embed_model, gemini_model, collection, verbose=True
        )

        print(f"\n{'='*60}")
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        print(f"\nSources:")
        for chunk in chunks:
            print(f"  → {chunk['metadata']['section']} (score: {chunk['score']})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Load embedding model
    embed_model = load_embedding_model(EMBEDDING_MODEL)

    # 2. Connect to ChromaDB
    collection = load_collection(CHROMA_DB_PATH, COLLECTION_NAME)

    # 3. Setup Gemini
    gemini_model = setup_gemini()

    # Choose mode:
    #   "demo"        → runs preset questions, good for live training
    #   "interactive" → type your own questions

    MODE = "interactive"   # change to "demo" for training presentation

    if MODE == "demo":
        run_demo(embed_model, gemini_model, collection)
    else:
        run_interactive(embed_model, gemini_model, collection)


if __name__ == "__main__":
    main()