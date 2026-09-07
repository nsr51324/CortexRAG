---
title: CortexRAG Medical RAG API
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
tags:
- rag
- medical-ai
- sentence-transformers
- faiss
- fastapi
- cross-encoder
- groq
---

# 🩺 CortexRAG - Advanced Medical RAG API

**CortexRAG** is a high-performance, domain-specific Retrieval-Augmented Generation (RAG) system engineered for medical and clinical question answering. It combines semantic vector search (FAISS + Sentence Transformers), cross-encoder re-ranking, and high-speed LLM inference (Groq / Llama) wrapped in a lightweight **FastAPI** REST interface.

---

## 🌟 Features

- **Semantic Embedding Engine**: `all-MiniLM-L6-v2` dense vector retrieval using FAISS indexing.
- **Precision Re-ranking**: Cross-encoder scoring (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for optimal document relevance.
- **Medical Synonym Expansion**: Context-aware synonym mapping for expanded search recall.
- **Ultra-Fast REST API**: Built on FastAPI with asynchronous request handling and Pydantic validation.
- **Cloudflare Tunnel Ready**: Zero-trust public exposure without complex firewall configuration.

---

## 🏗 System Architecture

```
                               ┌───────────────────────────┐
                               │     Client Request        │
                               └─────────────┬─────────────┘
                                             │ POST /query
                                             ▼
                               ┌───────────────────────────┐
                               │   FastAPI Web Server      │
                               └─────────────┬─────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │                                           │
                       ▼                                           ▼
         ┌───────────────────────────┐               ┌───────────────────────────┐
         │  Query Vectorization      │               │ Medical Synonym Expansion │
         │  (all-MiniLM-L6-v2)       │               └─────────────┬─────────────┘
         └─────────────┬─────────────┘                             │
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │     FAISS Vector Index    │
                               └─────────────┬─────────────┘
                                             │ Top-N Candidate Docs
                                             ▼
                               ┌───────────────────────────┐
                               │   Cross-Encoder Reranker  │
                               │  (ms-marco-MiniLM-L-6-v2) │
                               └─────────────┬─────────────┘
                                             │ Top-K Ranked Context
                                             ▼
                               ┌───────────────────────────┐
                               │   LLM Synthesis (Groq)    │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │    JSON API Response      │
                               └───────────────────────────┘
```

---

## 📁 Repository Structure

```
CortexRAG/
├── API_DEPLOYMENT_PLAN.md      # Step-by-step API & tunnel setup documentation
├── README.md                   # Hugging Face & GitHub Project Card
├── question_embeddings.npy     # Pre-computed dense embeddings matrix
├── questions.index             # Binary FAISS vector search index
├── notebooks/                  # Experimental notebooks & cleaning scripts
│   ├── Medical_RAG_Sytem.ipynb
│   └── rag_data_cleaning.ipynb
└── rag_model/                  # Core RAG engine configurations & resources
    ├── rag_config.json         # Search, score & model parameters
    ├── requirements.txt        # Python dependency specifications
    └── models/                 # Synonyms & model metadata
        └── medical_synonyms.json
```

---

## ⚡ Quick Start & Installation

### 1. Prerequisites
- Python 3.9+
- Pip package manager

### 2. Environment Setup
```bash
# Clone repository
git clone https://huggingface.co/spaces/YOUR_USERNAME/CortexRAG
cd CortexRAG

# Create virtual environment
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r rag_model/requirements.txt fastapi uvicorn pydantic
```

### 3. Environment Variables
Set your Groq API Key (or other LLM provider keys):
```bash
# Windows PowerShell
$env:GROQ_API_KEY="your_groq_api_key_here"

# Linux/macOS
export GROQ_API_KEY="your_groq_api_key_here"
```

---

## 🚀 Running the Local API

Start the server using `uvicorn`:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Interactive API Documentation (Swagger UI) is available at:
👉 **`http://127.0.0.1:8000/docs`**

---

## 🌐 Exposing Publicly via Cloudflare Tunnel

To expose your local FastAPI server securely to the internet without port forwarding:

1. Download [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/get-started/create-local-tunnel/).
2. Run the tunnel pointing to your local port:
   ```bash
   cloudflared tunnel --url http://127.0.0.1:8000
   ```
3. Use the generated URL (e.g. `https://xxx.trycloudflare.com`) as your public API endpoint.

---

## 🔌 API Reference & Integration Guide

### Endpoint
`POST /query`

### Request Headers
```http
Content-Type: application/json
```

### Request Payload Example
```json
{
  "question": "What are the first-line treatments for type 2 diabetes?",
  "top_k": 6
}
```

### Response Payload Example
```json
{
  "status": "success",
  "question": "What are the first-line treatments for type 2 diabetes?",
  "answer": "First-line pharmacological management for type 2 diabetes typically includes Metformin alongside lifestyle modifications...",
  "retrieved_context": [
    {
      "doc_id": 42,
      "text": "Metformin remains the initial drug of choice for monotherapy...",
      "rerank_score": 4.85
    }
  ],
  "execution_time_sec": 0.38
}
```

### Python Integration Example
```python
import requests

url = "https://your-cloudflare-url.trycloudflare.com/query"
payload = {
    "question": "What are the common causes of chest pain?",
    "top_k": 5
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

---

## 🛠 Configuration Parameters (`rag_config.json`)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `embedding_model` | `all-MiniLM-L6-v2` | SentenceTransformer embedding model |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precision reranking cross-encoder model |
| `retrieve_top_n` | `20` | Initial FAISS vector retrieval candidate count |
| `rerank_top_k` | `6` | Number of context snippets passed to LLM |
| `min_similarity_floor` | `0.4` | Cosine similarity threshold |

---

## 📜 License
This project is released under the **MIT License**.
