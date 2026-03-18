# Chunking Strategy

---

## What is Chunking?

Before storing documents in a vector database, you cannot store the entire document as one piece.
It is too large. The embedding model has a token limit. The search becomes inaccurate.

So you split the document into smaller pieces. Each piece is called a **chunk**.

**Real-time example:**

Imagine your HR policy is 50 pages.
Someone asks — *"What is the WFH policy?"*

If you embed all 50 pages as one chunk and search,
the result is noisy — everything is mixed together.

If you split each section into its own chunk,
the WFH section comes back as a clean, focused result.

Chunking = giving RAG a better chance to find the right answer.

---

## Why Chunking Strategy Matters

Bad chunking = bad answers. Even with a great embedding model.

```
SAME QUESTION → "How many sick leaves do I get?"

Bad chunk  →  "...leave encashment is calculated as Basic Salary
               divided by 26. Sick leave is 12 days. WFH is not
               a substitute for leave. Notice period for Band 1..."

Good chunk →  "Employees are entitled to 12 days of Sick Leave
               per calendar year. It cannot be carried forward
               or encashed. Medical certificate required after
               2 consecutive days."
```

The good chunk has one focused topic.
The bad chunk has four different topics mixed together.
RAG retrieves the whole chunk — so focused chunks = focused answers.

---

## Types of Chunking

### 1. Fixed Size Chunking

Split the document every N characters or words regardless of content.

```
Chunk 1 → words 1 to 200
Chunk 2 → words 201 to 400
Chunk 3 → words 401 to 600
```

**Problem:**
```
"...employees must serve a notice period of 60 days.
 ---- CHUNK BOUNDARY ----
 The resignation must be submitted via HRMS..."
```

Sentence is cut in the middle. Context is lost.
RAG gets an incomplete chunk and gives an incomplete answer.

Use only when document has no structure at all.

---

### 2. Fixed Size with Overlap

Same as fixed size but each chunk shares some content with the next one.

```
Chunk 1 → words 1  to 200
Chunk 2 → words 150 to 350   ← 50 words overlap
Chunk 3 → words 300 to 500   ← 50 words overlap
```

The overlap makes sure context is not lost at boundaries.
Better than plain fixed size but still not structure-aware.

---

### 3. Structure-Aware Chunking — Best for Markdown

Split based on the document's own structure — headings, sections, paragraphs.

```
# Section 4 — Leave Policy           → Parent chunk
## 4.1 Casual Leave                   → Child chunk 1
## 4.2 Sick Leave                     → Child chunk 2
## 4.3 Earned Leave                   → Child chunk 3
```

Each heading becomes a natural chunk boundary.
No sentence is cut in the middle.
Each chunk has exactly one topic.

**This is why Markdown is the best format for RAG.**
The structure is already there. You just follow it.

---

### 4. Semantic Chunking

Split not by size or structure but by meaning.
When the topic changes, start a new chunk.
Uses embedding similarity to detect topic shift.

```
Paragraph 1 talks about sick leave     → chunk continues
Paragraph 2 still talks about sick leave → chunk continues
Paragraph 3 shifts to WFH policy       → new chunk starts here
```

Most accurate but slow and expensive to compute.
Not recommended for training demos — too complex to explain quickly.

---

## Chunking Comparison

```
Strategy              | Structure | Accuracy | Complexity | Use When
----------------------|-----------|----------|------------|------------------
Fixed Size            | No        | Low      | Simple     | Plain text, no structure
Fixed Size + Overlap  | No        | Medium   | Simple     | Plain text, need safety net
Structure-Aware       | Yes       | High     | Medium     | Markdown, HTML, Word docs
Semantic              | Yes       | Highest  | Complex    | Production systems
```

---

## What We Will Use — Structure-Aware

Our document `technova_hr_policy.md` uses Markdown headings.

```
# 1. Company Overview
# 2. Employment Terms
# 3. Working Hours
# 4. Leave Policy
## 4.1 Casual Leave
## 4.2 Sick Leave
...
```

Our chunking rule is simple:

```
Every ## heading = one chunk
Chunk text = everything under that heading until the next heading
```

That gives us clean, focused, single-topic chunks.
Perfect for training. Easy to explain. Easy to debug.

---

## Chunk Size — How Big is Too Big?

```
Too small  →  "12 days of Sick Leave"
               Not enough context. Answer will be incomplete.

Too large  →  All of Section 4 in one chunk (500+ words)
               Too much noise. RAG retrieves everything.
               Accuracy drops.

Just right →  One sub-section per chunk (100 to 300 words)
               One topic. Enough context. Clean retrieval.
```

For our HR policy document, each `##` section is naturally 100 to 250 words.
No adjustment needed. Structure does the work for us.

---

## What a Final Chunk Looks Like

After chunking, each chunk will have two parts — content and metadata.

```
CHUNK 7

Content:
"Employees are entitled to 12 days of Sick Leave per calendar year.
 Sick Leave is credited in full on January 1st of each year.
 Any illness lasting more than two consecutive days requires a
 medical certificate from a registered doctor. Sick Leave cannot
 be availed immediately before or after a public holiday without
 medical certification. Unused Sick Leave lapses at year end and
 cannot be carried forward or encashed."

Metadata:
{
  "doc_name"      : "technova_hr_policy.md",
  "section"       : "4.2 Sick Leave",
  "parent_section": "4. Leave Policy",
  "chunk_index"   : 7,
  "heading_level" : 2
}
```

The content goes into the vector database for semantic search.
The metadata is used for filtering and showing the source in the answer.

---

## Chunking Flow

```
technova_hr_policy.md
          |
          |  Read file line by line
          ▼
    Is this line a ## heading?
    YES → save previous chunk, start new chunk
    NO  → keep adding lines to current chunk
          |
          ▼
    List of chunks
    [chunk1, chunk2, chunk3 ... chunk20]
          |
          |  Attach metadata to each chunk
          ▼
    [
      { content: "...", metadata: { section: "4.2 Sick Leave" } },
      { content: "...", metadata: { section: "4.3 Earned Leave" } },
      ...
    ]
          |
          ▼
    Ready for embedding → next step
```

---

## Key Takeaway for Training

> The quality of your RAG answers depends more on chunking strategy than on the embedding model or the LLM. Garbage chunks = garbage answers. Clean chunks = clean answers.

---

