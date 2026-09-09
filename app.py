import os
import re
import time
import json
import difflib
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ==========================================
# Paths & Default Configurations
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "questions.index")
DATASET_PATH = os.path.join(BASE_DIR, "notebooks", "AHD_english_cleaned.xlsx")
SYNONYMS_PATH = os.path.join(BASE_DIR, "rag_model", "models", "medical_synonyms.json")
CONFIG_PATH = os.path.join(BASE_DIR, "rag_model", "rag_config.json")

DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

# Load config if available
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        RAG_CONFIG = json.load(f)
else:
    RAG_CONFIG = {
        "embedding_model": "all-MiniLM-L6-v2",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "retrieve_top_n": 20,
        "rerank_top_k": 6,
    }

# Dynamic Global State
rag_resources: Dict[str, Any] = {}

SYSTEM_PROMPT = """
You are **Sugrly**, a medical RAG assistant specialized ONLY in:

* Diabetes
* Endocrinology
* Thyroid disorders
* Hormonal disorders
* Endocrine glands and related disorders
* Nutrition, glucose management, insulin, and complications DIRECTLY related to diabetes/endocrinology

You operate in **STRICT RAG MODE** for all medical questions.

==================================================
ASSISTANT IDENTITY
==================

Your name is **Sugrly**.

If the user asks who you are, what your name is, or what you can help with, answer briefly and naturally.

Example:
User: "Who are you?"
Assistant: "I'm Sugrly, a medical assistant specialized in diabetes and endocrinology."

User: "What can you help me with?"
Assistant: "I can help with questions related to diabetes, endocrinology, thyroid disorders, hormones, insulin, glucose management, and related nutrition."

Do NOT claim to be a doctor.
Do NOT claim to diagnose patients.

==================================================
CASUAL CONVERSATION
===================

Simple social or conversational messages are NOT medical questions and should NOT be treated as out-of-domain medical questions.

Examples include:

* "Hi"
* "Hello"
* "Hey"
* "Good morning"
* "Good evening"
* "How are you?"
* "What's up?"
* "Thanks"
* "Thank you"
* "Bye"
* "Goodbye"
* "Nice to meet you"

For these messages, respond naturally, briefly, and politely.

Examples:

User: "Hi"
Assistant: "Hi! I'm Sugrly. How can I help you?"

User: "Hello"
Assistant: "Hello! How can I help you today?"

User: "How are you?"
Assistant: "I'm doing well! How can I help you with diabetes or endocrinology?"

User: "Thanks"
Assistant: "You're welcome!"

User: "Bye"
Assistant: "Goodbye! Take care."

Do NOT retrieve medical evidence for simple greetings or casual conversation.

Do NOT respond with the out-of-domain refusal for casual conversation.

==================================================
CRITICAL DOMAIN RULE
====================

The **USER QUESTION itself** determines whether the question is in-domain.

A question is IN-DOMAIN only if its actual subject is:

* Diabetes
* Endocrinology
* Thyroid
* Hormones
* Endocrine glands/disorders
* Nutrition, glucose management, insulin
* A complication explicitly related to diabetes/endocrinology

If the user's actual question is about another body system or another medical specialty, it is OUT-OF-DOMAIN.

Do NOT classify a question as in-domain merely because the retrieved documents contain relevant medical words.

==================================================
TYPE 1 — IN-DOMAIN MEDICAL QUESTION
===================================

If AND ONLY IF the USER QUESTION itself is in-domain:

* Answer using ONLY the RETRIEVED MEDICAL EVIDENCE.
* Do NOT use pretrained/background medical knowledge.
* Do NOT guess.
* Do NOT invent missing details.
* Do NOT add medical facts that are not supported by the retrieved evidence.

If the retrieved evidence does not contain enough information to answer the question safely, classify it as insufficient evidence and follow the TYPE 3 response.

==================================================
DOSAGE & TREATMENT SAFETY — MANDATORY
=====================================

These rules OVERRIDE everything else for any answer involving:

* Medications
* Insulin
* Doses
* Units
* Quantities
* Treatment amounts

1. NEVER extract a numeric dosage from a retrieved document that describes a SPECIFIC PATIENT CASE and present it as a general recommendation.

2. A dosage found in evidence is ONLY valid for the exact clinical scenario described in that document.

3. If the user's question is GENERAL, such as:
   "How much insulin should I take?"

   and the evidence only contains case-specific dosages, DO NOT quote those numbers as a general answer.

4. For ANY medication dosage question where the evidence is:

   * Case-specific, OR
   * Contradicted by other evidence, OR
   * Insufficient to determine a safe dose for the user's exact situation

   explicitly state that the dose CANNOT be determined from the available information without individual physician assessment.

   Always direct the user to consult their treating physician.

5. Do NOT present partial evidence as a complete treatment recommendation.

Example — WRONG:

User:
"How much insulin should I take?"

Evidence:
"If sugar > 300, inject 4-8 units."

Wrong answer:
"You may need 4-8 units of insulin."

Example — CORRECT:

"Insulin doses cannot be determined from general guidelines. They depend on the patient's individual situation, and the appropriate dose cannot be determined from the available information. Please consult your treating physician."

==================================================
TYPE 2 — APP / IDENTITY QUESTIONS
=================================

For direct questions about Sugrly or the application itself:

* Answer briefly.
* Answer naturally.
* Do not perform medical retrieval unless the user also asks a medical question.

Examples:

"Who are you?"
→ "I'm Sugrly, a medical assistant specialized in diabetes and endocrinology."

"What is Sugrly?"
→ "Sugrly is a medical RAG assistant specialized in diabetes and endocrinology."

"What can you do?"
→ "I can help with diabetes, endocrinology, thyroid, hormones, insulin, glucose management, and related nutrition questions."

==================================================
TYPE 3 — OUT-OF-DOMAIN / INSUFFICIENT EVIDENCE
==============================================

For medical questions that are outside the supported specialty OR questions that are in-domain but cannot be safely answered from the retrieved evidence:

If the user wrote in Arabic, reply ONLY:

"معرفش، السؤال ده مش جزء من تخصصي (السكر والغدد الصماء)."

If the user wrote in English, reply ONLY:

"I don't know — this is outside my specialty (diabetes and endocrinology)."

Do NOT provide additional medical information in TYPE 3.

==================================================
RESPONSE PRIORITY
=================

Before answering, classify the user's message in this order:

1. **CASUAL CONVERSATION**

   * Greetings, thanks, goodbye, small talk.
   * Respond naturally.
   * No medical retrieval required.

2. **APP / IDENTITY**

   * Questions about Sugrly or the application.
   * Answer briefly and naturally.

3. **IN-DOMAIN MEDICAL**

   * Use ONLY retrieved medical evidence.
   * Follow all dosage and treatment safety rules.

4. **OUT-OF-DOMAIN MEDICAL**

   * Use the exact TYPE 3 refusal.

5. **IN-DOMAIN BUT INSUFFICIENT EVIDENCE**

   * Use the exact TYPE 3 refusal.

The user's actual question always determines the medical domain.

==================================================
FINAL SAFETY NOTE — TYPE 1 ONLY
===============================

At the end of every TYPE 1 medical answer, write exactly:

"This information is based on the available medical evidence and is for general informational purposes. It is not a diagnosis or a substitute for professional medical advice."
"""

# ==========================================
# RAG Helper Logic
# ==========================================
def expand_query(query: str, synonyms: dict) -> str:
    lower_q = query.lower()
    extra_terms = [
        med_term
        for phrase, med_term in synonyms.items()
        if phrase in lower_q and med_term not in lower_q
    ]
    if not extra_terms:
        return query
    return f"{query} ({', '.join(dict.fromkeys(extra_terms))})"

def vector_search(query: str, top_n: int = 20):
    embedder = rag_resources["embedder"]
    faiss_index = rag_resources["faiss_index"]
    metadata_store = rag_resources["metadata_store"]
    synonyms = rag_resources["synonyms"]

    query_for_embedding = expand_query(query, synonyms)
    q_emb = embedder.encode(
        [query_for_embedding],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, idxs = faiss_index.search(q_emb, top_n)
    
    retrieved = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        payload = metadata_store[int(idx)]
        retrieved.append({
            "doc_id": payload["doc_id"],
            "Question": payload["question"],
            "Answer": payload["answer"],
            "Category": payload["category"],
            "similarity": float(score),
        })
    return pd.DataFrame(retrieved)

def rerank(query: str, candidates: pd.DataFrame, top_k: int = 6, alpha: float = 0.6) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    reranker = rag_resources["reranker"]
    pairs = [
        (query, f"Question: {row['Question']}\nAnswer: {row['Answer']}")
        for _, row in candidates.iterrows()
    ]

    rerank_scores = reranker.predict(pairs)
    reranked = candidates.copy()
    reranked["rerank_score"] = rerank_scores

    def norm(s):
        s = s.astype(float)
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng > 0 else s * 0

    reranked["sim_norm"] = norm(reranked["similarity"])
    reranked["rerank_norm"] = norm(reranked["rerank_score"])
    reranked["final_score"] = alpha * reranked["rerank_norm"] + (1 - alpha) * reranked["sim_norm"]

    return reranked.sort_values("final_score", ascending=False).head(top_k).reset_index(drop=True)

def is_near_duplicate(text_a: str, text_b: str, threshold: float = 0.92) -> bool:
    return difflib.SequenceMatcher(None, text_a, text_b).ratio() > threshold

def build_evidence(reranked: pd.DataFrame, max_answer_chars: int = 700) -> List[dict]:
    evidence = []
    seen_answers = []

    for _, row in reranked.iterrows():
        answer = str(row["Answer"])
        if any(is_near_duplicate(answer, seen) for seen in seen_answers):
            continue
        seen_answers.append(answer)

        evidence.append({
            "doc_id": int(row["doc_id"]),
            "question": str(row["Question"]),
            "answer": answer[:max_answer_chars] + ("..." if len(answer) > max_answer_chars else ""),
            "category": str(row["Category"]),
            "similarity": round(float(row["similarity"]), 3),
            "rerank_score": round(float(row["rerank_score"]), 3),
        })
    return evidence

def generate_answer(query: str, user_data: str, evidence: List[dict]) -> str:
    groq_client = rag_resources["groq_client"]
    if not evidence:
        return "I don't know — this is outside my specialty (diabetes and endocrinology)."

    evidence_blocks = [
        f"[{e['doc_id']}] (category: {e['category']})\nRelated question: {e['question']}\nAnswer: {e['answer']}"
        for e in evidence
    ]
    evidence_text = "\n\n".join(evidence_blocks)

    user_message = f"""
USER QUESTION:
{query}

USER DATA:
{user_data}

RETRIEVED MEDICAL EVIDENCE:
{evidence_text}

TASK:
Answer the USER QUESTION using ONLY the RETRIEVED MEDICAL EVIDENCE for medical facts.
Do NOT cite document IDs in your answer.
If the evidence is insufficient, irrelevant, or only applies to specific patient cases
that differ from the user's question, give a clear refusal instead of guessing.
"""

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.1,
            max_tokens=1200,
        )
        msg = response.choices[0].message
        content = msg.content or getattr(msg, "reasoning", "") or ""
        content = content.strip()
        
        if not content:
            content = "I don't know — the available medical evidence is insufficient to answer this question accurately."

        answer = re.sub(r'\[\d+\]', '', content).strip()
        return answer   
    except Exception as e:
        # Fallback or error handling for Groq API
        return f"Error generating answer from LLM: {str(e)}"

# ==========================================
# FastAPI Lifecycle & App Initialization
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading RAG resources...")

    # Load FAISS index
    if not os.path.exists(FAISS_INDEX_PATH):
        raise RuntimeError(f"FAISS index file not found at: {FAISS_INDEX_PATH}")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)

    # Load Dataset for Metadata
    if not os.path.exists(DATASET_PATH):
        raise RuntimeError(f"Dataset file not found at: {DATASET_PATH}")
    df = pd.read_excel(DATASET_PATH, engine="openpyxl").reset_index(drop=True)
    df["doc_id"] = df.index
    metadata_store = [
        {
            "doc_id": row["doc_id"],
            "question": row["Question"],
            "answer": row["Answer"],
            "category": row["Category"],
        }
        for _, row in df.iterrows()
    ]

    # Load Synonyms
    synonyms = {}
    if os.path.exists(SYNONYMS_PATH):
        with open(SYNONYMS_PATH, "r", encoding="utf-8") as f:
            synonyms = json.load(f)

    # Load Transformer Models
    embedder = SentenceTransformer(RAG_CONFIG.get("embedding_model", "all-MiniLM-L6-v2"))
    reranker = CrossEncoder(RAG_CONFIG.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))

    # Load Groq Client
    groq_client = Groq(api_key=DEFAULT_GROQ_API_KEY)

    # Save to global state
    rag_resources["faiss_index"] = faiss_index
    rag_resources["metadata_store"] = metadata_store
    rag_resources["synonyms"] = synonyms
    rag_resources["embedder"] = embedder
    rag_resources["reranker"] = reranker
    rag_resources["groq_client"] = groq_client

    print(f"RAG resources loaded successfully! Index size: {faiss_index.ntotal}")
    yield
    print("Shutting down RAG resources.")
    rag_resources.clear()

class UTF8JSONResponse(JSONResponse):
    """Returns proper UTF-8 JSON — fixes PowerShell/Windows encoding display."""
    def render(self, content) -> bytes:
        import json
        return json.dumps(content, ensure_ascii=False, allow_nan=False).encode("utf-8")

app = FastAPI(
    title="CortexRAG Medical API",
    description="Production-ready FastAPI interface for CortexRAG Medical Question Answering Engine.",
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=UTF8JSONResponse
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    return UTF8JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": "Validation Error: Request payload does not match expected JSON schema.",
            "details": exc.errors()
        }
    )

# ==========================================
# Request / Response Schemas
# ==========================================
class QueryRequest(BaseModel):
    question: str = Field(..., example="What are the common symptoms of hypothyroidism?", min_length=2)
    user_data: Optional[str] = Field(default="", example="Age: 45, Gender: Female")
    top_k: Optional[int] = Field(default=6, ge=1, le=20)

class EvidenceItem(BaseModel):
    question: str
    answer: str
    category: str

class QueryResponse(BaseModel):
    status: str
    question: str
    answer: str
    execution_time_sec: float

# ==========================================
# Endpoints
# ==========================================
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "online",
        "service": "CortexRAG Medical API",
        "documentation": "/docs"
    }

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "healthy",
        "faiss_index_entries": rag_resources.get("faiss_index", None).ntotal if "faiss_index" in rag_resources else 0,
        "model_loaded": "embedder" in rag_resources
    }

def _run_rag(payload: QueryRequest) -> QueryResponse:
    start_time = time.time()
    try:
        candidates = vector_search(payload.question, top_n=RAG_CONFIG.get("retrieve_top_n", 20))
        reranked_df = rerank(payload.question, candidates, top_k=payload.top_k)
        evidence = build_evidence(reranked_df)
        answer = generate_answer(payload.question, payload.user_data or "", evidence)
        elapsed = round(time.time() - start_time, 3)
        return QueryResponse(
            status="success",
            question=payload.question,
            answer=answer,
            execution_time_sec=elapsed
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during inference: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse, tags=["RAG Inference"])
def query(payload: QueryRequest):
    return _run_rag(payload)

@app.post("/predict", response_model=QueryResponse, tags=["RAG Inference"])
def predict(payload: QueryRequest):
    return _run_rag(payload)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
