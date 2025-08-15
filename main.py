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

# OpenAI의 text-embedding-3-small 모델 차원
# 중요: PRD에 명시된 Gemini `embedding-001` 모델은 768 차원을 반환합니다.
# OpenAI 모델(1536 차원)과 차원이 달라 Faiss 인덱스에 함께 사용할 수 없습니다.
# 이 문제를 해결하려면 동일한 차원의 모델을 사용하거나, 별도의 인덱스를 관리해야 합니다.
# 여기서는 PRD의 요구사항을 따르되, OpenAI가 기본이라는 전제 하에 1536으로 설정합니다.
DIMENSION = 1536
FAISS_INDEX_PATH = "faq_index.faiss"
METADATA_PATH = "faq_metadata.json"

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
    top_k: int = Field(default=3, gt=0, le=10)

# --- Global Variables ---
# Faiss 인덱스와 메타데이터는 앱 생명주기 동안 관리
index = None
metadata_list = []

# --- Helper Functions ---
def load_data():
    """서버 시작 시 Faiss 인덱스와 메타데이터를 파일에서 로드합니다."""
    global index, metadata_list
    try:
        print(f"Loading Faiss index from {FAISS_INDEX_PATH}...")
        index = faiss.read_index(FAISS_INDEX_PATH)
        print("Faiss index loaded successfully.")
    except RuntimeError:
        print(f"Faiss index file not found. Initializing a new index with dimension {DIMENSION}.")
        index = faiss.IndexFlatL2(DIMENSION)

    try:
        print(f"Loading metadata from {METADATA_PATH}...")
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)
        print(f"Metadata loaded successfully. {len(metadata_list)} items.")
    except FileNotFoundError:
        print("Metadata file not found. Initializing an empty metadata list.")
        metadata_list = []

def save_data():
    """Faiss 인덱스와 메타데이터를 파일에 저장합니다."""
    print(f"Saving Faiss index to {FAISS_INDEX_PATH}...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("Faiss index saved.")

    print(f"Saving metadata to {METADATA_PATH}...")
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=4)
    print("Metadata saved.")

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
    """OpenAI를 우선 시도하고, 실패 시 Gemini로 fallback하는 안전한 임베딩 함수입니다."""
    try:
        return await get_openai_embedding(text)
    except Exception as e:
        print(f"[WARN] OpenAI embedding failed, falling back to Gemini: {e}")
        # Gemini 임베딩은 차원이 다르므로 Faiss에 추가 시 오류가 발생할 수 있습니다.
        # 이 예제에서는 PRD 요구사항을 보여주기 위해 호출하지만, 실제로는 차원 일치 작업이 필요합니다.
        raise HTTPException(
            status_code=500,
            detail=f"Primary embedding service failed, and fallback is not compatible due to dimension mismatch. OpenAI Error: {e}"
        )
        # return await get_gemini_embedding(text) # 실제 사용 시 주석 해제 및 차원 문제 해결 필요

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
    doc_id = str(payload.id)
    text = f"Q: {payload.question}\nA: {payload.answer}\n(ENG: {payload.translation or ''})"

    try:
        embedding = await get_safe_embedding(text)
        vec = np.array(embedding, dtype='float32').reshape(1, -1)

        # 기존에 동일한 ID가 있는지 확인하고, 있다면 업데이트 (여기서는 간단히 추가)
        index.add(vec)
        metadata_list.append({
            "id": doc_id,
            "category": payload.category,
            "subcategory": payload.subcategory,
            "text": text
        })

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
    """사용자 질문에 대해 Faiss에서 유사한 FAQ를 검색하여 반환합니다."""
    if index.ntotal == 0:
        return {"results": []}

    try:
        query_embedding = await get_safe_embedding(payload.question)
        query_vec = np.array(query_embedding, dtype='float32').reshape(1, -1)

        D, I = index.search(query_vec, payload.top_k)

        results = [metadata_list[i] for i in I[0]]
        return {"results": results}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"[ERROR] Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "RAG Faiss Server is running.", "indexed_items": index.ntotal}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)
