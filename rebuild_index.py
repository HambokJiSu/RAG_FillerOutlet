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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI의 text-embedding-3-small 모델 차원
DIMENSION = 1536
FAISS_INDEX_PATH = "faq_index.faiss"
METADATA_PATH = "faq_metadata.json"

# --- Embedding Function (from main.py) ---
async def get_openai_embedding(text: str):
    """OpenAI 임베딩을 가져옵니다."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        payload = {"model": "text-embedding-3-small", "input": text}
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]

async def get_safe_embedding(text: str):
    """OpenAI를 우선 시도하고, 실패 시 예외를 발생시키는 안전한 임베딩 함수입니다."""
    try:
        return await get_openai_embedding(text)
    except Exception as e:
        print(f"\n[ERROR] OpenAI embedding failed: {e}")
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
    
    for item in metadata_list:
        doc_id_str = item.get('id')
        question = item.get("question", "")
        answer = item.get("answer", "")
        translation = item.get("translation", "")

        if not doc_id_str:
            print(f"\n[WARN] Item with missing 'id' found. Skipping. Content: Q: {question[:20]}...")
            continue
        try:
            vector_ids.append(int(doc_id_str))
            texts_to_embed.append(f"Q: {question}\nA: {answer}\n(ENG: {translation or ''})")
        except (ValueError, TypeError):
            print(f"\n[WARN] Item with invalid 'id': {doc_id_str}. Skipping.")
            continue

    if not texts_to_embed:
        print("No valid items to process after filtering. An empty index will be created.")
        faiss.write_index(index, FAISS_INDEX_PATH)
        print(f"Created an empty index file: {FAISS_INDEX_PATH}")
        return

    print(f"Generating embeddings for {len(texts_to_embed)} items in batch...")
    try:
        tasks = [get_safe_embedding(text) for text in texts_to_embed]
        embeddings = await asyncio.gather(*tasks)
        
        vectors = np.array(embeddings, dtype='float32')
        ids_array = np.array(vector_ids, dtype='int64')

        print("Adding vectors to the Faiss index...")
        index.add_with_ids(vectors, ids_array)

    except Exception as e:
        print(f"\n[ERROR] An error occurred during embedding or index building: {e}")
        print("Stopping the rebuild process. The index file will not be saved.")
        return

    print(f"\nIndex build complete. Total vectors: {index.ntotal}")
    print(f"Saving new Faiss index to {FAISS_INDEX_PATH}...")
    faiss.write_index(index, FAISS_INDEX_PATH)
    print("Faiss index saved successfully.")
    print("Rebuild process finished.")

if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY environment variable not set. Please create a .env file or set it.")
    else:
        asyncio.run(main())
