"""
STEP 4: Hybrid Search (Vector Search + BM25 Keyword Search)
================================================================
ทำไมต้อง hybrid?
  - Vector search (semantic): เก่ง "ความหมายใกล้เคียง" เช่นถาม "ใบไหม้เป็นยังไง"
    หาเจอ "ใบแห้งเป็นหย่อมๆ" ได้ทั้งที่คำไม่ตรงกันเป๊ะ
    จุดอ่อน: คำเฉพาะเจาะจง (ชื่อสารเคมี, ชื่อวิทยาศาสตร์) อาจถูกลด priority
  - BM25 (keyword): เก่งจับคำตรงตัว เช่นถ้า user พิมพ์ "Phytophthora" มาตรงๆ
    จุดอ่อน: ไม่เข้าใจ paraphrase/ความหมายใกล้เคียงเลย
  -> รวมกันเพื่อชดเชยจุดอ่อนซึ่งกันและกัน

วิธีรวมผล: Reciprocal Rank Fusion (RRF)
  แทนที่จะเอา score ดิบจากสองระบบมาถ่วงน้ำหนักรวมกัน (ปัญหาคือ cosine
  similarity กับ BM25 score คนละ scale กัน เทียบกันตรงๆ ไม่ได้) RRF ใช้
  แค่ "อันดับ" (rank) ของแต่ละ chunk ในแต่ละระบบ:

      score(chunk) = sum( 1 / (k + rank_in_system) )  for each system

  k คือค่าคงที่ (ปกติ = 60) ทำหน้าที่ลด impact ของอันดับต้นๆ ไม่ให้ dominant
  เกินไป, chunk ที่ติดอันดับดีทั้งสองระบบจะได้คะแนนรวมสูงสุด
"""

import os
import json
import pickle

from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "durian_diseases"
BM25_PATH = "bm25_index.pkl"
EMBED_MODEL = "text-embedding-3-small"

RRF_K = 60


def embed_query(question: str) -> list[float]:
    response = client.embeddings.create(model=EMBED_MODEL, input=[question])
    return response.data[0].embedding


def simple_thai_tokenize(text: str) -> list[str]:
    """ต้องเหมือนกับ tokenizer ตอนสร้าง index ใน step 3 เป๊ะๆ ไม่งั้น BM25 เทียบผิด"""
    import re

    tokens = []
    for word in re.findall(r"[a-zA-Z0-9]+|[ก-๙]+", text):
        if re.match(r"[a-zA-Z0-9]+", word):
            tokens.append(word.lower())
        else:
            if len(word) <= 2:
                tokens.append(word)
            else:
                tokens.extend([word[i : i + 2] for i in range(len(word) - 1)])
    return tokens


def vector_search(question: str, top_k: int = 10) -> list[str]:
    """คืนค่า list ของ chunk id เรียงตาม similarity มากไปน้อย"""
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)

    query_embedding = embed_query(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["ids"][0]


def bm25_search(question: str, top_k: int = 10) -> list[str]:
    """คืนค่า list ของ chunk id เรียงตาม BM25 score มากไปน้อย"""
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)

    tokenized_query = simple_thai_tokenize(question)
    scores = data["bm25"].get_scores(tokenized_query)

    ranked = sorted(zip(data["ids"], scores), key=lambda x: x[1], reverse=True)
    return [chunk_id for chunk_id, score in ranked[:top_k] if score > 0]


def reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """
    รวมหลาย ranked list เป็น score เดียวด้วย RRF
    rank_lists: list ของ [chunk_id เรียงตามอันดับ] จากแต่ละระบบ (vector, bm25)
    """
    scores: dict[str, float] = {}
    for ranked_ids in rank_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def get_chunk_texts(chunk_ids: list[str]) -> dict[str, dict]:
    """ดึง text + metadata ของ chunk id ที่ระบุ (จาก BM25 pickle เพราะมีข้อมูลครบอยู่แล้ว)"""
    with open(BM25_PATH, "rb") as f:
        data = pickle.load(f)
    id_to_data = {
        cid: {"text": text, "page": meta["page"]}
        for cid, text, meta in zip(data["ids"], data["texts"], data["metadatas"])
    }
    return {cid: id_to_data[cid] for cid in chunk_ids if cid in id_to_data}


def hybrid_search(question: str, top_k: int = 5, candidate_k: int = 10) -> list[dict]:
    """
    Hybrid search หลัก: รัน vector search + bm25 search แยกกัน (ดึงมาคนละ
    candidate_k อันดับแรก) แล้วรวมด้วย RRF คืนค่า top_k อันดับสุดท้าย
    """
    vector_ids = vector_search(question, top_k=candidate_k)
    bm25_ids = bm25_search(question, top_k=candidate_k)

    fused = reciprocal_rank_fusion([vector_ids, bm25_ids])
    top_ids = [cid for cid, score in fused[:top_k]]

    chunk_data = get_chunk_texts(top_ids)

    results = []
    for cid, score in fused[:top_k]:
        if cid in chunk_data:
            results.append({
                "id": cid,
                "score": score,
                "page": chunk_data[cid]["page"],
                "text": chunk_data[cid]["text"],
            })
    return results


if __name__ == "__main__":
    test_question = "โรครากเน่าโคนเน่ามีอาการอย่างไร"
    print(f"คำถาม: {test_question}\n")

    results = hybrid_search(test_question, top_k=5)
    for r in results:
        print(f"[{r['id']}] page={r['page']} rrf_score={r['score']:.4f}")
        print(r["text"][:150] + "...")
        print()
