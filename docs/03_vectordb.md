# Vector Database

---

## What is a Vector Database?

After chunking and embedding, every chunk becomes a list of numbers (vector).
You need somewhere to store these vectors and search them fast.

A vector database does exactly two things:

```
1. Store   → save chunks with their vectors and metadata
2. Search  → given a query vector, find the closest chunk vectors
```

A normal database like MySQL stores rows and columns.
It searches by exact match — WHERE name = 'Rahul'.

A vector database stores numbers (vectors).
It searches by similarity — find me vectors closest in meaning to this query.

**Real-time example:**

```
You ask → "Can I take leave when my mother is sick?"

Normal DB search  → looks for exact words "mother" and "sick" in documents
                    Finds nothing useful

Vector DB search  → understands the meaning — personal emergency, family
                    Finds → Casual Leave section + Bereavement Leave section
                    Even though neither uses the word "mother"
```

That is the core difference. Meaning-based search vs word-based search.

---

## How Similarity Search Works

When you store a chunk, the vector database saves:

```
ID      → chunk_007
Vector  → [0.23, -0.87, 0.41, 0.99, -0.12 ... 384 numbers]
Content → "Employees get 12 days of Sick Leave per year..."
Metadata→ { section: "4.2 Sick Leave", doc: "hr_policy.md" }
```

When you search with a question:

```
Question → "How many sick leaves do I get?"
Embed    → [0.21, -0.91, 0.38, 0.95, -0.09 ... 384 numbers]

Vector DB compares this with ALL stored vectors
Finds    → chunk_007 is closest (sick leave section)
Returns  → content + metadata of top N closest chunks
```

The similarity measure used is called **Cosine Similarity**.
It measures the angle between two vectors.
Angle close to 0 = very similar meaning.
Angle close to 90 = very different meaning.

---

## Vector Databases Available

### 1. ChromaDB

```
Type        → Open source, runs locally
Setup       → pip install chromadb — that is it
Storage     → saves to a folder on your machine
Best for    → learning, prototyping, small projects
Limit       → not suitable for millions of vectors in production
Hosted      → yes, ChromaDB cloud is available
```

**Real-time example use case:**
Your RAG training project, personal projects, hackathons,
internal tools with less than 100k documents.

---

### 2. FAISS — Facebook AI Similarity Search

```
Type        → Open source library by Meta
Setup       → pip install faiss-cpu
Storage     → in-memory, you must save/load manually
Best for    → when you need raw speed and control
Limit       → no built-in metadata filtering, no persistence by default
              you have to manage everything yourself
```

**Real-time example use case:**
Research projects, when you want full control over the index,
when you are building a custom search engine from scratch.

---

### 3. Pinecone

```
Type        → Fully managed cloud vector database
Setup       → create account, get API key, pip install pinecone
Storage     → cloud hosted, no local storage
Best for    → production applications, large scale, teams
Limit       → paid after free tier (1 index, 100k vectors free)
              needs internet connection always
```

**Real-time example use case:**
Customer support chatbot for a company with 10k support documents,
e-commerce product search, enterprise RAG systems.

---

### 4. Weaviate

```
Type        → Open source, self-hosted or cloud
Setup       → Docker or Weaviate Cloud
Storage     → self-hosted or managed cloud
Best for    → production with advanced filtering needs
              supports hybrid search out of the box (vector + keyword)
Limit       → Docker setup adds complexity for beginners
```

**Real-time example use case:**
When you need both semantic search AND keyword filtering together.
Example — find resumes with Java experience (keyword) who also
match "microservices architect" (semantic).

---

### 5. Qdrant

```
Type        → Open source, self-hosted or cloud
Setup       → Docker or pip install qdrant-client
Storage     → self-hosted or Qdrant cloud
Best for    → production, very fast, good filtering support
              Rust-based so extremely performant
Limit       → slightly more setup than ChromaDB
```

**Real-time example use case:**
High-performance production RAG systems,
when you need filtering + semantic search at scale.

---

### 6. Milvus

```
Type        → Open source, enterprise grade
Setup       → Docker or Zilliz cloud (managed)
Storage     → self-hosted or cloud
Best for    → very large scale — billions of vectors
              used by companies like Salesforce, eBay
Limit       → heavy setup, overkill for small projects
```

**Real-time example use case:**
Large enterprise systems — searching across millions of product
listings, large-scale document intelligence platforms.

---

### 7. pgvector

```
Type        → PostgreSQL extension
Setup       → add extension to existing Postgres DB
Storage     → inside your existing PostgreSQL database
Best for    → teams already using PostgreSQL
              no new infrastructure needed
              combine vector search with relational queries
Limit       → slower than dedicated vector DBs at large scale
```

**Real-time example use case:**
You already have employee data in PostgreSQL.
You want to add semantic search without adding a new database.
Just add pgvector to existing Postgres — one database for everything.

---

## Comparison at a Glance

```
Database    | Open Source | Setup      | Scale          | Best For
------------|-------------|------------|----------------|---------------------------
ChromaDB    | Yes         | pip only   | Small-Medium   | Learning and prototyping
FAISS       | Yes         | pip only   | Medium-Large   | Custom search, research
Pinecone    | No (SaaS)   | API key    | Large          | Production, managed
Weaviate    | Yes         | Docker     | Large          | Hybrid search production
Qdrant      | Yes         | Docker     | Large          | High-performance production
Milvus      | Yes         | Docker     | Very Large     | Billions of vectors
pgvector    | Yes         | Postgres   | Medium         | Existing Postgres teams
```

---

## Which One to Use When

```
SITUATION                                      → USE THIS

Learning RAG for the first time               → ChromaDB
Hackathon or weekend project                  → ChromaDB
Production app, small team, managed           → Pinecone
Production app, self-hosted, cost saving      → Qdrant
Need vector + keyword search together         → Weaviate
Already on PostgreSQL, want to add search     → pgvector
Research, need raw speed and control          → FAISS
Enterprise, millions of documents             → Milvus
```

---

## For Our Training

We use **ChromaDB** because:

```
✓ One pip install — no Docker, no account, no API key
✓ Runs fully on laptop — works offline in training room
✓ Data persists to a folder — students can inspect it
✓ Simple Python API — easy to read and understand
✓ Supports metadata filtering — shows real concepts
✓ Good enough for our 20 chunk HR policy document
```

When you go to production → swap ChromaDB with Qdrant or Pinecone.
The RAG logic stays exactly the same. Only the database client changes.

---

## ChromaDB — How It Stores Data

```
chroma_db/                        ← folder on disk
  └── hr_policy_collection/
        ├── chunk_001
        │     vector   → [0.23, -0.87, 0.41...]
        │     content  → "Sick Leave is 12 days..."
        │     metadata → { section: "4.2 Sick Leave" }
        │
        ├── chunk_002
        │     vector   → [0.11, 0.54, -0.33...]
        │     content  → "WFH allowed 8 days per month..."
        │     metadata → { section: "5.2 WFH Entitlement" }
        │
        └── ... 20 chunks total
```

Everything saved locally. You can delete the folder and start fresh.
Students can see exactly what is stored. Great for teaching.

---

## Key Takeaway for Training

> A vector database is not magic. It is just a database that stores numbers and finds the closest numbers to your query. The magic is in the embedding — converting meaning into numbers. The database just stores and searches them fast.

---

