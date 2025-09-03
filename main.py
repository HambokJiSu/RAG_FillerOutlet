import os
import numpy as np
import faiss
import httpx
import json
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from contextlib import asynccontextmanager

# .env 파일에서 환경 변수 로드
load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# 로컬 임베딩 모델 서버 주소 (예: Ollama)
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://192.168.0.48:11434/api")

# 임베딩 모델과 생성 모델을 분리하여 정의
# LOCAL_EMBEDDING_MODEL_NAME = "nomic-embed-text" # 임베딩 전용 모델
LOCAL_EMBEDDING_MODEL_NAME = "dengcao/Qwen3-Embedding-0.6B:F16" # 임베딩 전용 모델
LOCAL_GENERATION_MODEL_NAME = "gpt-oss:20b"      # 답변 생성용 모델

# 중요: 사용하는 모델에 따라 차원(dimension)을 맞춰야 합니다.
# - OpenAI text-embedding-3-small: 1536
# - nomic-embed-text (Ollama): 768
# - dengcao/Qwen3-Embedding-8B (Ollama): 1024
# - dengcao/Qwen3-Embedding-0.6B (Ollama): 1024
# 여기서는 임베딩 모델인 dengcao/Qwen3-Embedding-0.6B를 기준으로 1024로 설정합니다.
DIMENSION = 1024
FAISS_INDEX_PATH = "faq_index.faiss"
METADATA_PATH = "faq_metadata.json"

# 유사도 임계값 (Squared L2 distance). 값이 작을수록 유사도가 높음.
# nomic-embed-text (normalized) 임베딩의 경우, (dist^2 = 2 - 2 * cos_sim) 입니다.
# 예: cos_sim 0.7 -> dist^2 0.6, cos_sim 0.8 -> dist^2 0.4
# 1.0은 cos_sim 0.5에 해당하며, 이보다 낮은 유사도는 관련성이 거의 없다고 판단합니다.
# 모델이 변경되었으므로 이 값은 테스트를 통해 조정이 필요할 수 있습니다.
SIMILARITY_THRESHOLD = 1.0

# --- Data Models ---
class WebhookPayload(BaseModel):
    id: int
    category: str | None = None
    subcategory: str | None = None
    question: str
    answer: str
    translation: str | None = None

class QueryPayload(BaseModel):
    question: str
    top_k: int = Field(default=3, gt=0, le=10) # LLM에게 더 많은 컨텍스트를 주기 위해 기본값을 3으로 변경

# --- Global Variables ---
# Faiss 인덱스와 메타데이터는 앱 생명주기 동안 관리
index = None
metadata_list = []
id_to_metadata = {} # ID를 메타데이터로 빠르게 조회하기 위한 맵

# --- Helper Functions ---
def load_data():
    """서버 시작 시 Faiss 인덱스와 메타데이터를 파일에서 로드합니다."""
    global index, metadata_list, id_to_metadata
    try:
        print(f"Loading Faiss index from {FAISS_INDEX_PATH}...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        if not hasattr(index, 'add_with_ids'):
            print("[ERROR] Loaded Faiss index does not support updates (not an IndexIDMap).")
            print("[ERROR] Please delete the old index file and restart the server to create a compatible one.")
            # 호환되지 않는 인덱스로는 실행을 중단하거나, 마이그레이션 로직을 추가해야 합니다.
            # 여기서는 에러를 출력하고 비정상 상태로 둘 수 있으므로, 실제 운영에서는 처리가 필요합니다.
            raise RuntimeError("Incompatible Faiss index type.")
        print("Faiss index loaded successfully.")
    except (RuntimeError, FileNotFoundError):
        print(f"Faiss index file not found or incompatible. Initializing a new index with dimension {DIMENSION}.")
        # 업데이트를 지원하려면 ID를 매핑할 수 있는 IndexIDMap을 사용해야 합니다.
        base_index = faiss.IndexFlatL2(DIMENSION)
        index = faiss.IndexIDMap(base_index)

    try:
        print(f"Loading metadata from {METADATA_PATH}...")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)
        # 메타데이터 로드 후, 빠른 조회를 위한 맵 생성
        id_to_metadata = {int(item['id']): item for item in metadata_list}
        print(f"Metadata loaded successfully. {len(metadata_list)} items.")
    except FileNotFoundError:
        print("Metadata file not found. Initializing an empty metadata list.")
        metadata_list = []
        id_to_metadata = {}

def save_data():
    """Faiss 인덱스와 메타데이터를 파일에 저장합니다."""
    print(f"Saving Faiss index to {FAISS_INDEX_PATH}...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("Faiss index saved.")

    print(f"Saving metadata to {METADATA_PATH}...")
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=4)
    print("Metadata saved.")

async def get_local_embedding(text: str):
    """자체 서버의 임베딩 모델을 호출합니다. (Ollama 기준)"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {"model": LOCAL_EMBEDDING_MODEL_NAME, "prompt": text}
        r = await client.post(f"{OLLAMA_API_BASE_URL}/embeddings", json=payload)
        r.raise_for_status()
        return r.json()["embedding"]

async def generate_answer_with_local_llm(question: str, context: str):
    """
    Ollama의 생성 모델(gpt-oss:20b)을 사용하여 질문과 컨텍스트 기반의 답변을 생성합니다.
    """
    prompt = f"""You are a helpful AI assistant for a company named 'Filler Outlet'. Your task is to answer user questions based ONLY on the provided 'Context' from the FAQ. Answer in Korean. If the context doesn't contain the answer, say that you cannot find the information.

Context:
---
{context}
---

Question: {question}

Answer:"""

    async with httpx.AsyncClient(timeout=120.0) as client:
        payload = {"model": LOCAL_GENERATION_MODEL_NAME, "prompt": prompt, "stream": False}
        r = await client.post(f"{OLLAMA_API_BASE_URL}/generate", json=payload)
        r.raise_for_status()
        return r.json()["response"]

async def get_openai_embedding(text: str):
    """OpenAI 임베딩을 가져옵니다."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        payload = {"model": "text-embedding-3-small", "input": text}
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]

async def get_gemini_embedding(text: str):
    """Gemini 임베딩을 가져옵니다."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Gemini API는 모델 차원이 다르므로, 실제 운영 시에는 차원 변환 또는 다른 모델 사용 필요
        url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedText?key={GEMINI_API_KEY}"
        payload = {"model": "models/embedding-001", "content": {"parts": [{"text": text}]}}
        r = await client.post(url, json=payload)
        r.raise_for_status()
        # Gemini 결과값은 'embedding' > 'value' 에 있습니다.
        return r.json()["embedding"]["value"]

async def get_safe_embedding(text: str):
    """로컬 임베딩을 우선 사용하고, 실패 시 예외를 발생시키는 안전한 임베딩 함수입니다."""
    try:
        return await get_local_embedding(text)
    except Exception as e:
        print(f"[ERROR] Local embedding failed: {e}")
        # 로컬 임베딩 서버에 문제가 생겼을 때 API가 중단되도록 예외를 발생시킵니다.
        raise HTTPException(
            status_code=500,
            detail=f"Local embedding service failed. Please check the local model server. Error: {e}"
        )

# --- FastAPI App Lifecycle ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 시 데이터 로드
    load_data()
    yield
    # 서버 종료 시 (선택적): 현재는 각 이벤트 후 저장하므로 필요 없음

app = FastAPI(lifespan=lifespan)

# --- API Endpoints ---
@app.post("/webhook")
async def webhook(payload: WebhookPayload):
    """Google Spreadsheet에서 받은 데이터로 Faiss 인덱스와 메타데이터를 업데이트합니다."""
    doc_id = payload.id # Pydantic 모델에 의해 이미 int
    # 요구사항: question 필드만으로 유사도를 비교하기 위해 question 텍스트만 사용합니다.
    # text = f"Q: {payload.question}\nA: {payload.answer}\n(ENG: {payload.translation or ''})"
    text = payload.question

    try:
        embedding = await get_safe_embedding(text)
        vec = np.array(embedding, dtype='float32').reshape(1, -1)
        vector_id_arr = np.array([doc_id], dtype='int64')

        new_metadata = {
            "id": doc_id,
            "category": payload.category,
            "subcategory": payload.subcategory,
            "question": payload.question,
            "answer": payload.answer,
            "translation": payload.translation
        }

        # ID 존재 여부를 확인하여 갱신 또는 추가
        if doc_id in id_to_metadata:
            print(f"Updating item with ID {doc_id}...")
            # 기존 벡터 제거 후 새 벡터 추가 (갱신)
            index.remove_ids(vector_id_arr)
            index.add_with_ids(vec, vector_id_arr)
            
            # 메타데이터 리스트에서 해당 아이템 찾아 갱신
            for i, item in enumerate(metadata_list):
                if int(item['id']) == doc_id:
                    metadata_list[i] = new_metadata
                    break
        else:
            print(f"Adding new item with ID {doc_id}...")
            # 새 벡터와 메타데이터 추가
            index.add_with_ids(vec, vector_id_arr)
            metadata_list.append(new_metadata)

        # 조회용 맵 갱신
        id_to_metadata[doc_id] = new_metadata

        # 변경 사항을 파일에 즉시 저장
        save_data()

        return {"status": "ok", "id": doc_id, "items_in_db": index.ntotal}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[ERROR] Webhook processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_rag(payload: QueryPayload):
    """사용자 질문에 대해 Faiss에서 유사한 FAQ를 검색하고, 그 결과를 바탕으로 LLM이 최종 답변을 생성하여 반환합니다."""
    print("\n--- New Query Received ---")
    print(f"Question: {payload.question}")
    print(f"Top K: {payload.top_k}")

    if index.ntotal == 0:
        print("[WARN] Index is empty. Cannot perform search.")
        return {
            "generated_answer": "데이터베이스가 비어있어 답변을 생성할 수 없습니다.",
            "source_documents": []
        }

    try:
        query_embedding = await get_safe_embedding(payload.question)
        query_vec = np.array(query_embedding, dtype='float32').reshape(1, -1)

        # 쿼리 벡터를 L2 정규화하여 검색 정확도를 보장합니다.
        faiss.normalize_L2(query_vec)

        # top_k만큼 유사한 문서를 검색
        print(f"\n1. Performing Faiss search with normalized query vector...")
        D, I = index.search(query_vec, payload.top_k)

        print("\n2. Initial search results (before filtering):")
        for i, (doc_id, dist) in enumerate(zip(I[0], D[0])):
            if doc_id == -1:
                print(f"  - Rank {i+1}: Invalid ID (-1).")
                continue
            question_preview = id_to_metadata.get(doc_id, {}).get('question', 'N/A')
            print(f"  - Rank {i+1}: ID={doc_id}, Distance={dist:.4f}, Q: '{question_preview[:50]}...'")

        # 유사도 임계값을 기준으로 결과 필터링
        source_documents = []
        print(f"\n3. Filtering results with threshold (Distance < {SIMILARITY_THRESHOLD}):")
        for doc_id, dist in zip(I[0], D[0]):
            if dist < SIMILARITY_THRESHOLD:
                if doc_id in id_to_metadata:
                    source_documents.append(id_to_metadata[doc_id])
                    print(f"  - [PASS] ID={doc_id}, Distance={dist:.4f}. Added to context.")
                else:
                    print(f"  - [WARN] ID={doc_id} passed threshold but not found in metadata.")
            else:
                print(f"  - [FAIL] ID={doc_id}, Distance={dist:.4f}. Discarded.")

        if not source_documents:
            # 일치하는 문서가 없을 때의 기본 답변
            print("\n4. No documents passed the similarity threshold.")
            generated_answer = "문의 내용과 관련된 정보를 찾지 못했습니다. 담당자가 직접 검토 후 회신 드릴 예정입니다."
            context_str = "No relevant context found."
        else:
            # LLM에 전달할 컨텍스트 생성
            context_str = "\n\n".join([
                f"Q: {doc['question']}\nA: {doc['answer']}" for doc in source_documents
            ])
            print("\n4. Final context for LLM:")
            print("--------------------")
            print(context_str)
            print("--------------------")
            
            # 로컬 LLM을 호출하여 답변 생성
            print("\n5. Generating answer with local LLM...")
            generated_answer = await generate_answer_with_local_llm(payload.question, context_str)
            print(f"\n6. Generated Answer: {generated_answer.strip()}")

        print("--- Query Process Finished ---\n")
        return {
            "generated_answer": generated_answer.strip(),
            "generated_answer": "임시",
            "source_documents": source_documents
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[ERROR] Query processing failed: {repr(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "RAG Faiss Server is running.", "indexed_items": index.ntotal}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
