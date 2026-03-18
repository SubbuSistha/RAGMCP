"""
chunk.py
-----------
Step 1 of RAG Pipeline — Read and Chunk

What this file does:
  - Reads the markdown document
  - Splits it into chunks based on ## headings
  - Attaches metadata to each chunk
  - Prints chunks so you can verify before embedding

Run:
  uv run python src/basic/chunk.py
"""

import json
import os


# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH = "kb/Policy.md"


# ── Step 1: Read the markdown file ───────────────────────────────────────────

def read_file(path: str) -> str:
    """Read the markdown file and return raw text."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    print(f"✓ File loaded: {path}")
    print(f"✓ Total characters: {len(content)}\n")
    return content


# ── Step 2: Split into chunks by ## heading ───────────────────────────────────

def chunk_by_heading(content: str, doc_name: str) -> list[dict]:
    """
    Split markdown content into chunks.

    Strategy:
      Every ## heading = one chunk
      Everything under that heading = chunk content
      # top-level headings = parent section tracker

    Returns:
      List of dicts with content and metadata
    """

    chunks      = []
    current_heading        = None
    current_parent         = None
    current_lines          = []
    chunk_index            = 0

    lines = content.split("\n")

    for line in lines:

        # Track top-level heading as parent section
        if line.startswith("# ") and not line.startswith("## "):
            # Save previous chunk before switching parent
            if current_heading and current_lines:
                chunk = build_chunk(
                    lines        = current_lines,
                    heading      = current_heading,
                    parent       = current_parent,
                    index        = chunk_index,
                    doc_name     = doc_name,
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1

            current_parent  = line.replace("# ", "").strip()
            current_heading = None
            current_lines   = []

        # Every ## heading starts a new chunk
        elif line.startswith("## "):
            # Save previous chunk before starting new one
            if current_heading and current_lines:
                chunk = build_chunk(
                    lines        = current_lines,
                    heading      = current_heading,
                    parent       = current_parent,
                    index        = chunk_index,
                    doc_name     = doc_name,
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1

            # Start new chunk
            current_heading = line.replace("## ", "").strip()
            current_lines   = []

        else:
            # Keep adding lines to current chunk
            if current_heading:
                current_lines.append(line)

    # Save the last chunk
    if current_heading and current_lines:
        chunk = build_chunk(
            lines    = current_lines,
            heading  = current_heading,
            parent   = current_parent,
            index    = chunk_index,
            doc_name = doc_name,
        )
        if chunk:
            chunks.append(chunk)

    return chunks


# ── Step 3: Build a single chunk with metadata ────────────────────────────────

def build_chunk(
    lines    : list[str],
    heading  : str,
    parent   : str,
    index    : int,
    doc_name : str,
) -> dict | None:
    """
    Build one chunk dict from lines and metadata.
    Returns None if content is empty after cleaning.
    """

    # Join lines and clean up extra whitespace
    content = "\n".join(lines).strip()

    # Skip empty chunks
    if not content:
        return None

    # Skip chunks that are just separators or metadata lines
    if len(content) < 30:
        return None

    return {
        "content"  : content,
        "metadata" : {
            "doc_name"      : doc_name,
            "section"       : heading,
            "parent_section": parent if parent else "General",
            "chunk_index"   : index,
            "word_count"    : len(content.split()),
        }
    }


# ── Step 4: Print chunks for verification ─────────────────────────────────────

def print_chunks(chunks: list[dict]) -> None:
    """Print all chunks so you can verify before embedding."""

    print("=" * 60)
    print(f"TOTAL CHUNKS CREATED: {len(chunks)}")
    print("=" * 60)

    for chunk in chunks:
        meta    = chunk["metadata"]
        content = chunk["content"]

        print(f"\n[Chunk {meta['chunk_index']}]")
        print(f"  Section : {meta['section']}")
        print(f"  Parent  : {meta['parent_section']}")
        print(f"  Words   : {meta['word_count']}")
        print(f"  Preview : {content[:120].replace(chr(10), ' ')}...")
        print("-" * 60)


# ── Step 5: Save chunks to JSON (optional, for inspection) ───────────────────

def save_chunks(chunks: list[dict], output_path: str) -> None:
    """Save chunks to a JSON file so you can inspect them."""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Chunks saved to: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # 1. Read the file
    content  = read_file(DATA_PATH)

    # 2. Chunk by heading
    doc_name = os.path.basename(DATA_PATH)
    chunks   = chunk_by_heading(content, doc_name)

    # 3. Print for verification
    print_chunks(chunks)

    # 4. Save to JSON for inspection
    save_chunks(chunks, "embedding/chunks.json")

    print(f"\n✓ Chunking complete. {len(chunks)} chunks ready for embedding.")


if __name__ == "__main__":
    main()