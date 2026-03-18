"""
consumer/chatbot.py
---------------
Streamlit Chatbot UI — TechNova HR Policy Assistant

What this file does:
  - Loads embedding model and ChromaDB once at startup
  - Provides a simple chat interface
  - Calls RAG pipeline from src/basic/query.py
  - Shows answer + sources used

Run:
  uv run streamlit run consumer/chatbot.py
"""

import sys
import os

# Add project root to path so we can import from src/
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from basic.query import (
    load_embedding_model,
    load_collection,
    setup_gemini,
    rag_query,
)


# ── Config ────────────────────────────────────────────────────────────────────

CHROMA_DB_PATH  = "chroma_db"
COLLECTION_NAME = "hr_policy"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title = "TechNova HR Assistant",
    page_icon  = "📋",
    layout     = "centered",
)


# ── Load models once using Streamlit cache ────────────────────────────────────
# @st.cache_resource runs only ONCE when the app starts
# Models stay in memory — no reload on every question

@st.cache_resource
def load_resources():
    """
    Load all heavy resources once at startup.
    Streamlit caches this — does not reload on every question.

    Returns embedding model, chromadb collection, gemini model.
    """
    embed_model  = load_embedding_model(EMBEDDING_MODEL)
    collection   = load_collection(CHROMA_DB_PATH, COLLECTION_NAME)
    gemini_model = setup_gemini()
    return embed_model, collection, gemini_model


# ── Header ────────────────────────────────────────────────────────────────────

st.title("📋 TechNova HR Policy Assistant")
st.caption("Ask any question about TechNova's HR and Leave Policy")
st.divider()


# ── Load resources ────────────────────────────────────────────────────────────

try:
    embed_model, collection, gemini_model = load_resources()
except FileNotFoundError as e:
    st.error(f"Setup incomplete: {e}")
    st.info("Run these first:\n```\nuv run python src/without_framework/01_chunk.py\nuv run python src/without_framework/02_embed_store.py\n```")
    st.stop()
except ValueError as e:
    st.error(f"Configuration error: {e}")
    st.stop()


# ── Chat history ──────────────────────────────────────────────────────────────
# st.session_state persists across reruns within the same session

if "messages" not in st.session_state:
    st.session_state.messages = []


# ── Display chat history ──────────────────────────────────────────────────────

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Sources used", expanded=False):
                for source in message["sources"]:
                    st.markdown(
                        f"**{source['metadata']['section']}**  "
                        f"`score: {source['score']}`  \n"
                        f"*{source['metadata']['parent_section']}*"
                    )


# ── Chat input ────────────────────────────────────────────────────────────────

question = st.chat_input("Ask your HR question here...")

if question:
    # Show user message
    with st.chat_message("user"):
        st.markdown(question)

    # Save user message to history
    st.session_state.messages.append({
        "role"   : "user",
        "content": question,
    })

    # Get RAG answer
    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents..."):
            answer, chunks = rag_query(
                question     = question,
                embed_model  = embed_model,
                gemini_model = gemini_model,
                collection   = collection,
                verbose      = False,   # no console output in UI mode
            )

        # Show answer
        st.markdown(answer)

        # Show sources in expander
        with st.expander("Sources used", expanded=False):
            for chunk in chunks:
                st.markdown(
                    f"**{chunk['metadata']['section']}**  "
                    f"`score: {chunk['score']}`  \n"
                    f"*{chunk['metadata']['parent_section']}*"
                )

    # Save assistant message to history
    st.session_state.messages.append({
        "role"   : "assistant",
        "content": answer,
        "sources": chunks,
    })


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("About")
    st.markdown("""
    This assistant answers questions about
    **TechNova HR & Leave Policy** using RAG.

    **How it works:**
    1. Your question is embedded into a vector
    2. ChromaDB finds the closest policy sections
    3. Gemini reads those sections and answers

    **Tech Stack:**
    - `sentence-transformers` — embedding
    - `chromadb` — vector search
    - `gemini-1.5-flash` — answer generation
    """)

    st.divider()

    st.header("Try these questions")
    sample_questions = [
        "How many sick leaves do I get?",
        "What is the WFH policy for managers?",
        "Can I carry forward earned leave?",
        "What is the notice period for senior managers?",
        "What happens during probation?",
    ]

    for q in sample_questions:
        st.markdown(f"- {q}")

    st.divider()

    # Clear chat button
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()