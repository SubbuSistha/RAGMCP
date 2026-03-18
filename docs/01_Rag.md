# RAG — Retrieval Augmented Generation

---

## What is RAG?

LLMs like Claude or ChatGPT are trained on data up to a certain date.
They do not know your company's data. They do not know what changed last week.

**RAG solves this.**

RAG lets you connect an LLM to your own documents so it can answer questions using your data — not just its training data.

**Real-time example:**

You ask ChatGPT — *"What is TechNova's leave policy?"*
ChatGPT says — *"I don't know TechNova."*

You add RAG with TechNova's HR document.
Now it says — *"TechNova gives 12 days of Sick Leave per year as per Section 4.2."*

That is RAG. Your data + LLM power combined.

---

## What is Embedding?

Embedding is converting text into numbers so a computer can understand meaning.

A sentence like *"I am sick today"* becomes an array of numbers like:
```
[0.23, -0.87, 0.41, 0.99, -0.12 .... 384 numbers]
```

These numbers capture the **meaning** of the sentence — not just the words.

**Real-time example:**

These three sentences have different words but similar meaning:

```
"I am not feeling well"
"I have fever and cold"
"I need to take sick leave"
```

Embedding places all three **close to each other** in vector space
because they all mean the same thing — illness.

When you search *"employee is unwell"* — the embedding finds all three
even though none of them contain the word *"unwell"*.

That is the power of embedding over simple keyword search.

---

## Why Embedding — Real-Time Example

Imagine you are HR and you have 500 policy documents.

An employee asks — *"Can I work from home when my child is sick?"*

**Without embedding (keyword search):**
```
Searches for → "work from home" + "child" + "sick"
Finds        → nothing useful because no document
               uses all three words together
Result       → No answer found
```

**With embedding (semantic search):**
```
Understands  → employee needs WFH for personal/family reason
Finds        → WFH policy section + Leave policy section
Result       → "You can apply WFH or Casual Leave for personal
               emergencies including family health situations."
```

Embedding understands **intent**, not just words.
That is why every RAG system starts with embedding.

---

## How RAG Works — Flow

```
YOUR DOCUMENT (HR Policy, Manual, Wiki)
          |
          |  Step 1 — CHUNKING
          |  Split document into small pieces
          |  Each piece = one topic or section
          |
          ▼
     [Chunk 1]  [Chunk 2]  [Chunk 3] ... [Chunk N]
     Sick Leave  WFH Policy  Exit Policy
          |
          |  Step 2 — EMBEDDING
          |  Convert each chunk into a vector (numbers)
          |  Store in Vector Database
          |
          ▼
     VECTOR DATABASE  (ChromaDB / Pinecone)
     [0.23, -0.87...]  ← Sick Leave chunk
     [0.11,  0.54...]  ← WFH Policy chunk
     [0.76, -0.22...]  ← Exit Policy chunk


          USER ASKS A QUESTION
          "How many sick leaves do I get?"
          |
          |  Step 3 — EMBED THE QUESTION
          |  Question also becomes a vector
          |  [0.21, -0.91, 0.38 ...]
          |
          ▼
          SIMILARITY SEARCH
          Compare question vector with all chunk vectors
          Find top 3 closest chunks
          |
          ▼
          RETRIEVED CHUNKS
          → Sick Leave section (most relevant)
          → Leave summary section
          → Probation leave section
          |
          |  Step 4 — AUGMENT + GENERATE
          |  Send question + retrieved chunks to LLM
          |
          ▼
          LLM (Claude / ChatGPT)
          Reads the retrieved chunks
          Answers the question
          |
          ▼
          FINAL ANSWER
          "You are entitled to 12 days of Sick Leave
           per calendar year as per Section 4.2."
```

---

## RAG Limitations

### 1. Misses what is not written
If the document says *"HDFC Bank, loan processing, KYC"*
and you ask *"Who has fintech experience?"*
RAG may miss it because the word *fintech* is not there.

### 2. Bad at counting and listing
Question — *"How many leave types does TechNova have?"*
RAG finds relevant chunks but cannot count accurately.
It needs a structured database for this.

### 3. Chunking breaks context
If one policy spans two chunks and gets split in the middle,
the answer may be incomplete or wrong.

### 4. Table and structure problem
RAG reads tables as flat text and loses the row-column relationship.
*"What is the increment for rating 4?"* may return wrong data
if the table is not handled properly during chunking.

### 5. Hallucination still happens
RAG reduces hallucination but does not eliminate it.
If the retrieved chunk is slightly off-topic,
the LLM may still generate a confident wrong answer.

---

## When to Use RAG

**Use RAG when:**

- Your document is too large to fit in the LLM context window
- You have 50 plus pages of unstructured text
- Questions are open-ended and need understanding of meaning
- Examples — HR policy bot, legal document Q&A, product manual assistant, internal wiki search

**Do NOT use RAG when:**

- You have less than 20 pages — just send everything to the LLM directly
- Questions need exact counts or filtering — use a database instead
- Data changes every minute — RAG needs re-indexing, not suitable for real-time data
- Questions are always exact keyword matches — simple search works fine

---

## One Line Summary

> RAG = Find the right information first, then let the LLM answer using that information.

---
