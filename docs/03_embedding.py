# Embeddings — Concepts, Best Practices and Libraries

---

## Quick Recap — What is an Embedding?

Embedding converts text into a fixed-size list of numbers that captures meaning.

```
"Employee is on sick leave"  →  [0.23, -0.87, 0.41, 0.99 ... 384 numbers]
"Staff member called in ill" →  [0.21, -0.91, 0.38, 0.95 ... 384 numbers]
"What is the WFH policy?"   →  [0.11,  0.54, -0.33, 0.12 ... 384 numbers]
```

First two are close in vector space — same meaning.
Third is far — different topic.

This is what makes semantic search possible.

---

## Two Ways to Generate Embeddings

### Way 1 — Local Model (runs on your machine)

```
Text → Embedding Model (downloaded locally) → Vector
```

No API call. No cost. No internet needed after first download.
Model runs on your CPU or GPU.

Best for: training demos, offline use, cost-sensitive projects.

### Way 2 — API Based (cloud call)

```
Text → HTTP Request → Cloud API → Vector returned
```

No model download. Pay per token or per request.
Always latest model. Needs internet.

Best for: production apps, when you need highest quality.

---

## How to Choose the Right Embedding Model

Ask these questions before choosing:

```
Is my text in English only?          → any model works
Is my text multilingual?             → use multilingual model
Do I need it to run offline?         → use local model
Is cost a concern?                   → use local model
Do I need highest accuracy?          → use API based model
How long is my text per chunk?       → check model token limit
Am I doing code search?              → use code-specific model
```

---

## Python — Embedding Libraries

---

### 1. sentence-transformers

```
Type      : Local model, runs on your machine
Install   : pip install sentence-transformers
Cost      : Free
Dimension : 384 (MiniLM) to 1024 (mpnet)
Best for  : RAG prototypes, training demos, offline use
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# Single text
vector = model.encode("How many sick leaves do I get?")
print(vector.shape)   # (384,)

# Batch — faster than one by one
texts = [
    "Sick leave policy",
    "Work from home rules",
    "Notice period details",
]
vectors = model.encode(texts)
print(vectors.shape)   # (3, 384)
```

**Popular models inside sentence-transformers:**

```
all-MiniLM-L6-v2          → 384 dims, fast, good quality   ← use for training
all-mpnet-base-v2          → 768 dims, slower, better quality
paraphrase-multilingual    → 768 dims, 50+ languages
multi-qa-MiniLM-L6-cos-v1 → 384 dims, optimised for Q&A search
```

---

### 2. OpenAI Embeddings

```
Type      : API based
Install   : pip install openai
Cost      : $0.00002 per 1000 tokens (very cheap)
Dimension : 1536 (ada-002) or 3072 (text-embedding-3-large)
Best for  : production apps needing high accuracy
```

```python
from openai import OpenAI

client = OpenAI(api_key="your_key")

response = client.embeddings.create(
    input = "How many sick leaves do I get?",
    model = "text-embedding-3-small",   # cheaper, good quality
)

vector = response.data[0].embedding
print(len(vector))   # 1536
```

**OpenAI models:**
```
text-embedding-3-small  → 1536 dims, cheap, good quality
text-embedding-3-large  → 3072 dims, best quality, more expensive
text-embedding-ada-002  → 1536 dims, older model, still widely used
```

---

### 3. Google Generative AI Embeddings

```
Type      : API based
Install   : pip install google-generativeai
Cost      : Free tier available (generous limits)
Dimension : 768
Best for  : when you already use Gemini for generation
            keeps everything in Google ecosystem
```

```python
import google.generativeai as genai

genai.configure(api_key="your_google_api_key")

result = genai.embed_content(
    model   = "models/text-embedding-004",
    content = "How many sick leaves do I get?",
)

vector = result["embedding"]
print(len(vector))   # 768
```

**Why this matters for your project:**
You are already using Google Gemini for answer generation.
Using Google embedding too keeps everything in one ecosystem.
One API key. One billing account.

---

### 4. HuggingFace Transformers (raw)

```
Type      : Local model
Install   : pip install transformers torch
Cost      : Free
Dimension : depends on model
Best for  : when you want full control, research, custom models
```

```python
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModel.from_pretrained(model_name)

text   = "How many sick leaves do I get?"
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)

# Mean pooling to get sentence vector
vector = outputs.last_hidden_state.mean(dim=1).squeeze()
print(vector.shape)   # torch.Size([384])
```

sentence-transformers is a wrapper around this.
It hides all the tokenizer and pooling complexity.
Use raw HuggingFace only when you need custom behaviour.

---

### 5. Cohere Embeddings

```
Type      : API based
Install   : pip install cohere
Cost      : Free tier, then paid
Dimension : 1024
Best for  : multilingual, strong on search and retrieval tasks
```

```python
import cohere

co     = cohere.Client("your_api_key")
result = co.embed(
    texts      = ["How many sick leaves do I get?"],
    model      = "embed-english-v3.0",
    input_type = "search_query",   # important — tell it this is a query
)

vector = result.embeddings[0]
print(len(vector))   # 1024
```

**Cohere has two modes — important concept:**
```
input_type = "search_document"  → use when embedding document chunks
input_type = "search_query"     → use when embedding user questions
```
Using the right input type improves accuracy significantly.

---

## Python Libraries — Comparison

```
Library              | Type    | Cost    | Dims | Best For
---------------------|---------|---------|------|-------------------------
sentence-transformers| Local   | Free    | 384+ | RAG demos, offline, fast
OpenAI               | API     | Low     | 1536 | Production, high accuracy
Google GenAI         | API     | Free+   | 768  | Google ecosystem projects
HuggingFace raw      | Local   | Free    | Any  | Research, full control
Cohere               | API     | Free+   | 1024 | Multilingual, retrieval
```

---

## Java — Embedding Libraries

---

### 1. LangChain4j — Most Popular

```
Type    : Framework with embedding support
Maven   : langchain4j-open-ai or langchain4j-embeddings
Cost    : Free (framework), API cost depends on provider
Best for: Java RAG applications, enterprise Java projects
```

```xml
<!-- pom.xml -->
<dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j-open-ai</artifactId>
    <version>0.30.0</version>
</dependency>
```

```java
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.data.embedding.Embedding;

EmbeddingModel model = OpenAiEmbeddingModel.builder()
    .apiKey("your_openai_key")
    .modelName("text-embedding-3-small")
    .build();

Embedding embedding = model.embed("How many sick leaves do I get?").content();
float[]   vector    = embedding.vector();

System.out.println("Dimensions: " + vector.length);   // 1536
```

LangChain4j also supports local models, Cohere, and more.
Swap the model class — rest of your code stays the same.

---

### 2. LangChain4j — Local Embedding (no API key)

```java
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.data.embedding.Embedding;

// Runs fully local — no API key needed
// Same all-MiniLM-L6-v2 model as Python sentence-transformers
EmbeddingModel model = new AllMiniLmL6V2EmbeddingModel();

Embedding embedding = model.embed("How many sick leaves do I get?").content();
float[]   vector    = embedding.vector();

System.out.println("Dimensions: " + vector.length);   // 384
```

```xml
<!-- pom.xml — for local ONNX model -->
<dependency>
    <groupId>dev.langchain4j</groupId>
    <artifactId>langchain4j-embeddings-all-minilm-l6-v2</artifactId>
    <version>0.30.0</version>
</dependency>
```

This is the Java equivalent of `sentence-transformers` in Python.
Same model, same vector dimensions, runs locally.

---

### 3. Spring AI

```
Type    : Spring ecosystem embedding support
Maven   : spring-ai-openai-spring-boot-starter
Cost    : Free (framework), API cost for provider
Best for: Spring Boot applications
```

```xml
<!-- pom.xml -->
<dependency>
    <groupId>org.springframework.ai</groupId>
    <artifactId>spring-ai-openai-spring-boot-starter</artifactId>
    <version>1.0.0</version>
</dependency>
```

```java
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.beans.factory.annotation.Autowired;

@Service
public class EmbeddingService {

    @Autowired
    private EmbeddingModel embeddingModel;

    public float[] embed(String text) {
        return embeddingModel.embed(text);
    }
}
```

```yaml
# application.yml
spring:
  ai:
    openai:
      api-key: your_key_here
      embedding:
        model: text-embedding-3-small
```

Spring AI handles the boilerplate. Good for existing Spring Boot projects.

---

### 4. Deep Java Library — DJL

```
Type    : AWS open source ML library for Java
Maven   : ai.djl
Cost    : Free
Best for: when you want pure Java ML without Python dependency
          runs HuggingFace models natively in Java
```

```xml
<dependency>
    <groupId>ai.djl.huggingface</groupId>
    <artifactId>tokenizers</artifactId>
    <version>0.27.0</version>
</dependency>
```

```java
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.huggingface.tokenizers.Encoding;

HuggingFaceTokenizer tokenizer =
    HuggingFaceTokenizer.newInstance("sentence-transformers/all-MiniLM-L6-v2");

Encoding encoding = tokenizer.encode("How many sick leaves do I get?");
long[]   inputIds = encoding.getIds();

// Pass to model for inference
```

DJL is more low level. LangChain4j is easier for most use cases.

---

## Java Libraries — Comparison

```
Library       | Type    | Cost   | Best For
--------------|---------|--------|----------------------------------
LangChain4j   | Both    | Free+  | RAG apps, easy API, most popular
Spring AI     | API     | Free+  | Spring Boot projects
DJL           | Local   | Free   | Pure Java ML, AWS ecosystem
```

---

## For Your Project — Recommendation

```
PYTHON side (what we are building):
  Use sentence-transformers        ← already set up
  If you switch to Google later:
  Use google-generativeai embed    ← same API key as Gemini

JAVA side (if someone in training asks):
  Use LangChain4j local model      ← same all-MiniLM-L6-v2
  No API key needed
  Vectors are compatible with Python vectors
  Can share the same ChromaDB
```

---

## Important — Consistency Rule

> Always use the SAME embedding model for storing and searching.
> If you embed chunks with `all-MiniLM-L6-v2` → you MUST search with `all-MiniLM-L6-v2`.
> Mixing models gives wrong results even if dimensions match.

```
WRONG:
  Store chunks  → all-MiniLM-L6-v2    (384 dims)
  Search query  → text-embedding-3-small (1536 dims)
  Result        → dimension mismatch error or garbage results

RIGHT:
  Store chunks  → all-MiniLM-L6-v2    (384 dims)
  Search query  → all-MiniLM-L6-v2    (384 dims)
  Result        → accurate semantic search
```

---

