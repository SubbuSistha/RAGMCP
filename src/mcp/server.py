"""
mcp/server.py
-------------
MCP Server — TechNova HR Policy RAG Tool

Exposes one MCP tool:
  search_hr_policy(question) → answer from RAG pipeline

Transport: HTTP (SSE) — works with Copilot, Claude Desktop, any MCP client

Run:
  uv run python src/mcp/server.py

Connect from Copilot:
  .vscode/mcp.json → { "url": "http://localhost:8000/mcp" }
"""

import os
import sys

# Add project root to path so we can import RAG pipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

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
CHROMA_DB_PATH  = "chroma_db_langchain_mcp"
COLLECTION_NAME = "hr_policy_langchain_mcp"
EMBEDDING_MODEL = "gemini-embedding-2-preview"
GEMINI_MODEL    = "gemini-3-flash-preview"
TOP_K           = 3
HOST            = "0.0.0.0"
PORT            = 8000


# ── FastMCP server ────────────────────────────────────────────────────────────

mcp = FastMCP(
    name         = "TechNova HR Policy Assistant",
    instructions = "Use search_hr_policy to answer questions about TechNova HR and Leave policy.",
    host         = HOST,
    port         = PORT,
)


# ── RAG components ────────────────────────────────────────────────────────────

def build_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model          = EMBEDDING_MODEL,
        google_api_key = os.getenv("GOOGLE_API_KEY"),
    )


def build_llm():
    return ChatGoogleGenerativeAI(
        model          = GEMINI_MODEL,
        google_api_key = os.getenv("GOOGLE_API_KEY"),
        temperature    = 0,
    )


def build_prompt():
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


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[Source: {doc.metadata.get('sub_section', doc.metadata.get('section', 'General'))}]\n"
        f"{doc.page_content}"
        for doc in docs
    )


def get_vectorstore():
    """
    Load existing ChromaDB.
    If not found → build it first from the markdown document.
    """

    embeddings = build_embeddings()

    if not os.path.exists(CHROMA_DB_PATH):
        print("ChromaDB not found. Building vector store first...")
        _build_vectorstore(embeddings)

    vectorstore = Chroma(
        persist_directory  = CHROMA_DB_PATH,
        embedding_function = embeddings,
        collection_name    = COLLECTION_NAME,
    )

    print(f"✓ Vector store loaded ({vectorstore._collection.count()} chunks)")
    return vectorstore


def _build_vectorstore(embeddings):
    # Read raw text directly
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#",  "section"),
            ("##", "sub_section"),
        ],
        strip_headers=False,
    )

    chunks = splitter.split_text(content)   # pass raw text directly

    Chroma.from_documents(
        documents         = chunks,
        embedding         = embeddings,
        persist_directory = CHROMA_DB_PATH,
        collection_name   = COLLECTION_NAME,
    )
    print(f"✓ Built vector store with {len(chunks)} chunks")

def build_rag_chain(vectorstore):
    """Build LCEL chain — same as with_framework version."""

    retriever = vectorstore.as_retriever(
        search_type   = "similarity",
        search_kwargs = {"k": TOP_K},
    )

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

    return chain


# ── Load RAG chain once at startup ────────────────────────────────────────────
# Loaded once when server starts — not on every tool call

print("Initialising RAG pipeline...")
vectorstore = get_vectorstore()
rag_chain   = build_rag_chain(vectorstore)
print("✓ RAG pipeline ready\n")


# ── MCP Tool ──────────────────────────────────────────────────────────────────

@mcp.tool()
def search_hr_policy(question: str) -> str:
    """
    Search TechNova HR and Leave Policy and return a grounded answer.

    Use this tool when the user asks anything about:
      - Leave policy (sick leave, casual leave, earned leave)
      - Work from home policy
      - Resignation and notice period
      - Compensation and benefits
      - Code of conduct
      - Performance management
      - Any TechNova HR related question

    Args:
        question: The HR policy question from the user

    Returns:
        Answer grounded in the HR policy document with source section
    """

    print(f"\n[MCP Tool Called]")
    print(f"  Question: {question}")

    result  = rag_chain.invoke(question)
    answer  = result["answer"]
    sources = result["sources"]

    # Append source references to answer
    source_refs = "\n".join(
        f"- {doc.metadata.get('sub_section', doc.metadata.get('section', 'General'))}"
        for doc in sources
    )

    full_response = f"{answer}\n\nSources:\n{source_refs}"

    print(f"  Answer : {answer[:80]}...")
    print(f"  Sources: {len(sources)} chunks used\n")

    return full_response


# ── Run server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Starting MCP server on http://{HOST}:{PORT}")
    print(f"Copilot config → http://localhost:{PORT}/mcp\n")
    mcp.run(transport="sse")