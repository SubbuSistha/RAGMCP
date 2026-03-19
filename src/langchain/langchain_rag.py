"""
01_langchain_rag.py
-------------------
RAG Pipeline — Full LangChain LCEL

Uses:
  SystemMessagePromptTemplate  → explicit system message
  HumanMessagePromptTemplate   → explicit human message

Dependencies:
  uv add langchain
  uv add langchain-google-genai
  uv add langchain-chroma
  uv add langchain-community
  uv add langchain-text-splitters
  uv add unstructured

Run:
  uv run python src/langchain/langchain_rag.py
"""

import os
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters             import MarkdownHeaderTextSplitter
from langchain_google_genai               import GoogleGenerativeAIEmbeddings
from langchain_google_genai               import ChatGoogleGenerativeAI
from langchain_chroma                     import Chroma
from langchain_core.prompts               import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers        import StrOutputParser
from langchain_core.runnables             import RunnablePassthrough, RunnableParallel

load_dotenv()


# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH       = "kb/Policy.md"
CHROMA_DB_PATH  = "chroma_db_langchain"
COLLECTION_NAME = "hr_policy_langchain"
EMBEDDING_MODEL = "gemini-embedding-2-preview"
GEMINI_MODEL    = "gemini-3-flash-preview"
TOP_K           = 3


# ── Components ────────────────────────────────────────────────────────────────

def build_embeddings():
    """Google embedding model — used for storing and searching."""
    return GoogleGenerativeAIEmbeddings(
        model          = EMBEDDING_MODEL,
        google_api_key = os.getenv("GOOGLE_API_KEY"),
    )


def build_llm():
    """Google Gemini — answer generation."""
    return ChatGoogleGenerativeAI(
        model          = GEMINI_MODEL,
        google_api_key = os.getenv("GOOGLE_API_KEY"),
        temperature    = 0,
    )


def build_prompt():
    """
    ChatPromptTemplate using explicit message classes.

    SystemMessagePromptTemplate → sets LLM role and instructions
    HumanMessagePromptTemplate  → carries the user question

    {context}  → auto filled with retrieved chunks
    {question} → user question passed through the chain
    """

    system_template = SystemMessagePromptTemplate.from_template(
        """You are an HR assistant for TechNova Solutions.
Answer the employee's question using ONLY the context provided below.
If the answer is not in the context, say "I could not find this information in the HR policy document."
Always mention which section your answer is from.

Context:
{context}"""
    )

    human_template = HumanMessagePromptTemplate.from_template("{question}")

    return ChatPromptTemplate.from_messages([
        system_template,
        human_template,
    ])


# ── Document loading and chunking ─────────────────────────────────────────────

def load_and_split(path: str):
    """
    UnstructuredMarkdownLoader → loads file as Document
    MarkdownHeaderTextSplitter → splits by ## headings
                                  auto attaches heading as metadata
    """

    print(f"Loading: {path}")

    loader = UnstructuredMarkdownLoader(path)
    docs   = loader.load()

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on = [
            ("#",  "section"),
            ("##", "sub_section"),
        ],
        strip_headers = False,
    )

    chunks = splitter.split_text(docs[0].page_content)

    print(f"✓ {len(chunks)} chunks created")
    print(f"\nSample chunk metadata:")
    for chunk in chunks[:3]:
        print(f"  → {chunk.metadata}")
    print()

    return chunks


# ── Vector store ──────────────────────────────────────────────────────────────

def get_vectorstore(chunks=None):
    """
    First run  → Chroma.from_documents() embeds + stores all chunks
    Later runs → Chroma() loads from disk, no re-embedding
    """

    embeddings = build_embeddings()

    if chunks:
        print("Building vector store...")
        vectorstore = Chroma.from_documents(
            documents         = chunks,
            embedding         = embeddings,
            persist_directory = CHROMA_DB_PATH,
            collection_name   = COLLECTION_NAME,
        )
        print(f"✓ {vectorstore._collection.count()} chunks stored\n")
    else:
        print("Loading existing vector store...")
        vectorstore = Chroma(
            persist_directory  = CHROMA_DB_PATH,
            embedding_function = embeddings,
            collection_name    = COLLECTION_NAME,
        )
        print(f"✓ {vectorstore._collection.count()} chunks loaded\n")

    return vectorstore


# ── Format retrieved docs ─────────────────────────────────────────────────────

def format_docs(docs) -> str:
    """
    Retriever returns list of Document objects.
    Format into a single string for the prompt context.
    """
    return "\n\n".join(
        f"[Source: {doc.metadata.get('sub_section', doc.metadata.get('section', 'General'))}]\n"
        f"{doc.page_content}"
        for doc in docs
    )


# ── Build LCEL chain ──────────────────────────────────────────────────────────

def build_chain(vectorstore):
    """
    Full LCEL chain using pipe operator.

    question (string)
        ↓
    RunnableParallel
        ├── context  → retriever | format_docs
        └── question → RunnablePassthrough
        ↓
    ChatPromptTemplate (SystemMessage + HumanMessage)
        ↓
    ChatGoogleGenerativeAI
        ↓
    StrOutputParser
        ↓
    answer (string)
    """

    retriever = vectorstore.as_retriever(
        search_type   = "similarity",
        search_kwargs = {"k": TOP_K},
    )

    # Full RAG chain — answer + sources in parallel
    chain = RunnableParallel({
        "answer": (
            RunnableParallel({
                "context" : retriever | format_docs,
                "question": RunnablePassthrough(),
            })
            | build_prompt()
            | build_llm()
            | StrOutputParser()
        ),
        "sources": retriever,
    })

    print("✓ LCEL chain ready\n")
    return chain


# ── Ask ───────────────────────────────────────────────────────────────────────

def ask(chain, question: str, verbose: bool = True) -> str:
    """Invoke the LCEL chain and display answer with sources."""

    if verbose:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print(f"{'='*60}")

    result  = chain.invoke(question)
    answer  = result["answer"]
    sources = result["sources"]

    if verbose:
        print(f"\nSOURCES RETRIEVED:")
        print("-" * 60)
        for i, doc in enumerate(sources):
            section = doc.metadata.get(
                "sub_section", doc.metadata.get("section", "N/A")
            )
            print(f"  [{i+1}] {section}")
            print(f"       {doc.page_content[:120].replace(chr(10), ' ')}...")
        print("-" * 60)
        print(f"\nANSWER:")
        print("=" * 60)
        print(answer)

    return answer


# ── Demo ──────────────────────────────────────────────────────────────────────

def run_demo(chain):
    """Preset questions for training. Mix of good and bad RAG questions."""

    questions = [
        "How many sick leaves do I get per year?",
        "What is the work from home policy for managers?",
        "What happens when I resign from the company?",
        "Can I carry forward my earned leave?",
        "How many total employees work at TechNova?",  # bad RAG — show limitation
    ]

    for question in questions:
        ask(chain, question)
        input("\nPress Enter for next question...")


# ── Interactive ───────────────────────────────────────────────────────────────

def run_interactive(chain):
    """Type your own questions. Type exit to quit."""

    print("\n" + "=" * 60)
    print("INTERACTIVE MODE — TechNova HR Policy Assistant")
    print("Type 'exit' to quit")
    print("=" * 60)

    while True:
        question = input("\nYour question: ").strip()

        if question.lower() in ("exit", "quit", "q"):
            print("Exiting.")
            break

        if not question:
            continue

        ask(chain, question)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    first_run = not os.path.exists(CHROMA_DB_PATH)

    if first_run:
        print("=== FIRST RUN — Building vector store ===\n")
        chunks      = load_and_split(DATA_PATH)
        vectorstore = get_vectorstore(chunks=chunks)
    else:
        print("=== Loading existing vector store ===\n")
        vectorstore = get_vectorstore()

    chain = build_chain(vectorstore)

    MODE = "interactive"   # change to "demo" for training

    if MODE == "demo":
        run_demo(chain)
    else:
        run_interactive(chain)


if __name__ == "__main__":
    main()
