# 📄 PRD: 구글 스프레드시트 기반 RAG 시스템 (Faiss 버전)

## 1. 개요
본 시스템은 **구글 스프레드시트**를 데이터 소스로 하여, 데이터 변경이 발생할 때마다 **Webhook 이벤트**를 통해 RAG 서버로 전달하고, **OpenAI Embedding API**를 우선적으로 활용하여 벡터를 **Faiss**에 적재한다.  
만약 OpenAI Embedding API 사용 시 장애(쿼터 초과, 네트워크 오류 등)가 발생하면 **Gemini Embedding API**를 fallback으로 사용하여 안정적인 운영을 보장한다.  

이를 통해 최신 데이터를 기반으로 한 FAQ 질의응답 시스템을 구현한다.

---

## 2. 목표
- 구글 스프레드시트의 **FAQ 데이터 변경사항을 실시간 반영**
- **벡터 검색 기반 RAG 시스템**으로 FAQ 검색 정확도 향상
- OpenAI 장애 발생 시 **자동으로 Gemini로 대체** → 서비스 안정성 확보
- Faiss를 활용하여 **대용량 벡터 검색** 최적화

---

## 3. 주요 요구사항

### 3.1 데이터 소스
- **구글 스프레드시트**
  - 주요 컬럼: `id`, `category`, `subcategory`, `question`, `answer`, `translation`
  - 관리자가 직접 수정 가능
  - 변경 발생 시 **Webhook** 호출

### 3.2 RAG 서버
- **FastAPI 기반 REST 서버**
- 주요 엔드포인트:
  - `POST /webhook`
    - 스프레드시트 변경 이벤트 수신
    - Embedding 생성 후 Faiss 인덱스 및 Metadata 업데이트
  - `POST /query`
    - 질문 입력 시 top-K 유사 FAQ 반환

### 3.3 Embedding
- **기본 엔진**: OpenAI `text-embedding-3-small`
- **Fallback 엔진**: Gemini `embedding-001`
- **안정성 로직**:
  - OpenAI 호출 실패 시 Gemini Embedding 자동 호출
  - 실패 원인은 로그에 남김

### 3.4 벡터 검색
- **Faiss** 인덱스 사용 (IndexFlatL2)
- Metadata는 별도 리스트 또는 DB와 연동
- 검색 시:
  1. 질문 Embedding 생성 (OpenAI → Gemini fallback)
  2. Faiss Top-K 검색
  3. Metadata 매핑 후 반환

---

## 4. 시스템 아키텍처

flowchart LR
    A[Google Spreadsheet] -->|Webhook Trigger| B[FastAPI Webhook]
    B -->|Embedding 요청 (OpenAI 우선)| C[OpenAI API]
    B -->|Fallback 요청| D[Gemini API]
    C -->|임베딩 결과| E[Faiss Index]
    D -->|임베딩 결과| E[Faiss Index]
    E -->|질의 응답 시 벡터 검색| F[Query Engine]
    F -->|결과 반환| G[사용자]

## 5. 소스코드 예제
from fastapi import FastAPI, Request
import numpy as np
import faiss
import httpx
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI()

dimension = 1536
index = faiss.IndexFlatL2(dimension)
metadata_list = []

async def get_openai_embedding(text: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        payload = {"model": "text-embedding-3-small", "input": text}
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]

async def get_gemini_embedding(text: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedText?key={GEMINI_API_KEY}"
        payload = {"text": text}
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()["embedding"]["value"]

async def get_safe_embedding(text: str):
    try:
        return await get_openai_embedding(text)
    except Exception as e:
        print(f"[WARN] OpenAI embedding 실패 → Gemini fallback: {e}")
        return await get_gemini_embedding(text)

@app.post("/webhook")
async def webhook(req: Request):
    data = await req.json()
    doc_id = str(data["id"])
    text = f"Q: {data['question']}\nA: {data['answer']}\n(ENG: {data['translation']})"
    embedding = await get_safe_embedding(text)
    vec = np.array(embedding, dtype='float32').reshape(1, -1)
    index.add(vec)
    metadata_list.append({
        "id": doc_id,
        "category": data.get("category"),
        "subcategory": data.get("subcategory"),
        "text": text
    })
    return {"status": "ok", "id": doc_id}

@app.post("/query")
async def query_rag(req: Request):
    payload = await req.json()
    question = payload["question"]
    top_k = payload.get("top_k", 3)
    query_embedding = await get_safe_embedding(question)
    query_vec = np.array(query_embedding, dtype='float32').reshape(1, -1)
    D, I = index.search(query_vec, top_k)
    results = [metadata_list[i] for i in I[0]]
    return {"results": results}

## 6. 비기능 요구사항
### 6.1 성능

### 6.2 안정성

### 6.3 보완

## 7. 운영 시나리오
### 1 관리자가 스프레드시트에서 FAQ 수정

### 2 구글 스프레드시트가 Webhook 호출 (/webhook)

### 3 서버가 문서 텍스트(Q+A+번역) 생성

### 4 OpenAI Embedding 요청

### 5 성공 → Faiss 저장

### 6 실패 → Gemini Embedding → Faiss 저장

### 7 사용자가 질문 입력

### 8 안전한 Embedding 생성 → Faiss Top-K 검색 → Metadata 매핑

### 9 최적 FAQ 반환

## 8. 로그 및 모니터링
- OpenAI 실패 시 → [WARN] OpenAI embedding 실패 → Gemini fallback
- Webhook 수신 ID, Embedding 호출 상태, fallback 여부 로깅

## 9. 향후 확장 계획
- OpenAI와 Gemini 임베딩 동시 저장 → 품질 비교 가능
- Redis 캐시로 자주 조회되는 질문 응답 속도 개선
- 관리자용 대시보드: Webhook 처리 현황, DB 상태, fallback 발생률 모니터링