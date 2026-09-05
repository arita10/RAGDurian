"""
STEP 3: Embedding + เก็บลง Vector Database (+ เตรียม BM25 index)
====================================================================
Embedding model ที่เลือก: OpenAI text-embedding-3-small (1536 dimensions)
  เหตุผล: เร็ว ถูก ($0.02/1M tokens) และคุณภาพเพียงพอสำหรับ retrieval ทั่วไป
  ถ้าทดสอบแล้วผลลัพธ์ไม่แม่นพอ ค่อย upgrade เป็น text-embedding-3-large ทีหลังได้
  (แค่เปลี่ยนชื่อ model แล้ว re-embed ใหม่ทั้งหมด)

Vector DB: Chroma (local, ไม่ต้องตั้ง server แยก เหมาะกับ prototype)

เพิ่มจากโค้ดเดิม: สร้าง BM25 index ควบคู่ไปด้วย (เก็บเป็น .pkl)
  BM25 = keyword-based search แบบดั้งเดิม (term frequency + inverse doc frequency)
  เก่งเรื่องหาคำเฉพาะเจาะจงที่ vector search อาจพลาด เช่น:
    - ชื่อวิทยาศาสตร์ (Phytophthora palmovora)
    - ชื่อสารเคมี (เมตาแลกซิล, คอปเปอร์ออกซีคลอไรด์)
  เพราะ vector search เน้น "ความหมายใกล้เคียง" ไม่ใช่ "คำตรงตัว"
  -> จะเอาไปใช้ทำ Hybrid Search ใน step 4

Output:
  - chroma_db/  (persistent vector store)
  - bm25_index.pkl (สำหรับ keyword search)
"""

import os
import json
import pickle

from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from rank_bm25 import BM25Okapi

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

CHUNKS_PATH = "durian_chunks.json"
CHROMA_DIR = "chroma_db"
BM25_PATH = "bm25_index.pkl"
COLLECTION_NAME = "durian_diseases"
EMBED_MODEL = "text-embedding-3-small"


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """เรียก OpenAI embedding API เป็น batch (ประหยัด request มากกว่าเรียกทีละอัน)"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
        print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return all_embeddings


def simple_thai_tokenize(text: str) -> list[str]:
    """
    Tokenizer ง่ายๆ สำหรับ BM25 (ภาษาไทยไม่มีเว้นวรรคคำ จึงต้อง tokenize เอง)
    ใช้วิธี character-bigram แทนการตัดคำแบบเต็มรูปแบบ (ไม่ต้องพึ่ง pythainlp)
    -> จับคู่ตัวอักษรที่ติดกันเป็นคู่ๆ ช่วยให้ BM25 เทียบคำที่คล้ายกันได้
       แม้จะไม่ตัดคำถูกความหมาย 100% แต่ทำงานได้ดีพอสำหรับ keyword matching เบื้องต้น
    ส่วนคำภาษาอังกฤษ/ตัวเลข จะ tokenize แบบคำปกติ (split ตาม whitespace)
    """
    import re

    tokens = []
    for word in re.findall(r"[a-zA-Z0-9]+|[ก-๙]+", text):
        if re.match(r"[a-zA-Z0-9]+", word):
            tokens.append(word.lower())
        else:
            # thai word -> character bigrams
            if len(word) <= 2:
                tokens.append(word)
            else:
                tokens.extend([word[i : i + 2] for i in range(len(word) - 1)])
    return tokens


def main():
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunked_docs = json.load(f)

    texts = [d["text"] for d in chunked_docs]
    ids = [d["id"] for d in chunked_docs]
    metadatas = [{"page": d["page"]} for d in chunked_docs]

    print(f"กำลัง embed {len(texts)} chunks ด้วย {EMBED_MODEL} ...")
    embeddings = embed_texts(texts)

    # --- Vector store (Chroma) ---
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    print(f"เก็บลง Chroma เรียบร้อย: {collection.count()} chunks")

    # --- BM25 index (สำหรับ hybrid search step ถัดไป) ---
    tokenized_corpus = [simple_thai_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)

    with open(BM25_PATH, "wb") as f:
        pickle.dump(
            {"bm25": bm25, "ids": ids, "texts": texts, "metadatas": metadatas},
            f,
        )
    print(f"สร้าง BM25 index เรียบร้อย: บันทึกที่ {BM25_PATH}")


if __name__ == "__main__":
    main()
