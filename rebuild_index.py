import os
import json
import numpy as np
import faiss
import httpx
from dotenv import load_dotenv
import asyncio

# --- Configuration ---
# .env 파일에서 환경 변수 로드
load_dotenv()
 
# 로컬 임베딩 모델 설정
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "http://192.168.0.48:11434/api")
LOCAL_EMBEDDING_MODEL_NAME = "dengcao/Qwen3-Embedding-0.6B:F16" # 임베딩 전용 모델
 
# Qwen3-Embedding-0.6B 모델의 차원은 1024입니다.
DIMENSION = 1024
FAISS_INDEX_PATH = "faq_index.faiss"
METADATA_PATH = "faq_metadata.json"

# --- Embedding Function (from main.py) ---
async def get_local_embedding(text: str):
    """자체 서버의 임베딩 모델을 호출합니다. (Ollama 기준)"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        payload = {"model": LOCAL_EMBEDDING_MODEL_NAME, "prompt": text}
        r = await client.post(f"{OLLAMA_API_BASE_URL}/embeddings", json=payload)
        r.raise_for_status()
        return r.json()["embedding"]

async def get_safe_embedding(text: str):
    """로컬 임베딩을 시도하고, 실패 시 예외를 발생시키는 안전한 임베딩 함수입니다."""
    try:
        return await get_local_embedding(text)
    except Exception as e:
        # 오류 발생 시 더 상세한 정보를 출력합니다.
        print(f"\n[ERROR] Local embedding failed for text: '{text[:100]}...'")
        print(f"[ERROR] Exception details: {repr(e)}")
        # 임베딩 실패 시 스크립트를 중지시키기 위해 예외를 다시 발생시킵니다.
        raise

# --- Main Rebuild Logic ---
async def main():
    """
    faq_metadata.json 파일을 읽어 ID 기반의 faq_index.faiss 파일을 새로 생성하거나 덮어씁니다.
    """
    print(f"Loading metadata from {METADATA_PATH}...")
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata_list = json.load(f)
        print(f"Metadata loaded successfully. Found {len(metadata_list)} items.")
    except FileNotFoundError:
        print(f"[ERROR] Metadata file not found at {METADATA_PATH}. Cannot proceed.")
        return
    except json.JSONDecodeError:
        print(f"[ERROR] Failed to parse {METADATA_PATH}. Please check if it's a valid JSON file.")
        return

    if not metadata_list:
        print("Metadata is empty. Creating an empty index.")
        if os.path.exists(FAISS_INDEX_PATH):
            os.remove(FAISS_INDEX_PATH)
            print(f"Removed existing index file: {FAISS_INDEX_PATH}")
        # 빈 인덱스라도 생성하여 일관성 유지
        base_index = faiss.IndexFlatL2(DIMENSION)
        index = faiss.IndexIDMap(base_index)
        faiss.write_index(index, FAISS_INDEX_PATH)
        print(f"Created an empty index file: {FAISS_INDEX_PATH}")
        return

    print(f"Initializing a new Faiss index (IndexIDMap) with dimension {DIMENSION}.")
    base_index = faiss.IndexFlatL2(DIMENSION)
    index = faiss.IndexIDMap(base_index)

    print("Preparing data for embedding")
    texts_to_embed = []
    vector_ids = []
    processed_ids = set()
    
    for item in metadata_list:
        doc_id_str = item.get('id')
        question = item.get("question", "")
        answer = item.get("answer", "")
        translation = item.get("translation", "")
        
        if not doc_id_str:
            print(f"\n[WARN] Item with missing 'id' found. Skipping. Content: Q: {question[:20]}...")
            continue
            
        text_to_embed = f"Q: {question}\nA: {answer}\n(ENG: {translation or ''})"
        # 임베딩할 텍스트가 비어있는지 확인합니다.
        if not text_to_embed.strip():
            print(f"\n[WARN] Item with empty content found for ID {doc_id_str}. Skipping.")
            continue

        try:
            # float 형태의 ID (e.g., 1.0)도 처리하기 위해 float으로 먼저 변환 후 int로 캐스팅
            vector_id = int(float(doc_id_str))
            
            # 중복된 ID가 있는지 확인하고 경고를 출력합니다.
            if vector_id in processed_ids:
                print(f"\n[WARN] Duplicate ID '{vector_id}' found. This can lead to inconsistent data.")
            processed_ids.add(vector_id)

            vector_ids.append(vector_id)
            texts_to_embed.append(text_to_embed)
        except (ValueError, TypeError, AttributeError):
            print(f"\n[WARN] Item with invalid or non-numeric 'id': {doc_id_str}. Skipping.")
            continue

    if not texts_to_embed:
        print("No valid items to process after filtering. The index file will not be created.")
        faiss.write_index(index, FAISS_INDEX_PATH)
        print(f"Created an empty index file: {FAISS_INDEX_PATH}")
        return

    print(f"Generating embeddings for {len(texts_to_embed)} items sequentially...")
    embeddings = []
    try:
        for i, text in enumerate(texts_to_embed):
            # 로컬 서버 과부하를 막기 위해 하나씩 순차적으로 처리합니다.
            embedding = await get_safe_embedding(text)
            embeddings.append(embedding)
            # 진행 상황을 10개 단위로 출력하여 사용자가 인지할 수 있도록 합니다.
            if (i + 1) % 10 == 0 or (i + 1) == len(texts_to_embed):
                print(f"  ... processed {i + 1}/{len(texts_to_embed)} items")
        
        vectors = np.array(embeddings, dtype='float32')
        # 임베딩된 벡터들을 L2 정규화하여 일관된 거리 계산을 보장합니다.
        faiss.normalize_L2(vectors)

        ids_array = np.array(vector_ids, dtype='int64')

        print("Adding normalized vectors to the Faiss index...")
        index.add_with_ids(vectors, ids_array)
    except Exception as e:
        # 오류 발생 시 더 상세한 정보를 출력합니다.
        print(f"\n[ERROR] An error occurred during embedding or index building: {repr(e)}")
        print("Stopping the rebuild process. The index file will not be saved.")
        return

    print(f"\nIndex build complete. Total vectors: {index.ntotal}")
    print(f"Saving new Faiss index to {FAISS_INDEX_PATH}...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("Faiss index saved successfully.")
    print("Rebuild process finished.")

if __name__ == "__main__":
    print("Starting index rebuild process using local embedding model...")
    asyncio.run(main())
