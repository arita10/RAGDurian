"""
STEP 6: Generation + Caching
================================
Generation:
  เอา chunk ที่ rerank แล้ว (context ตรงประเด็นที่สุด) ใส่ prompt แล้วสั่ง LLM
  ตอบโดย "อ้างอิงจาก context เท่านั้น" ไม่ให้เดาจากความรู้ทั่วไป (กัน hallucination)
  และให้บอกอ้างอิงหน้าด้วย เพื่อให้ user ตรวจสอบย้อนกลับไปที่เอกสารต้นฉบับได้

Caching: มี 2 จุดหลักที่ควร cache ใน RAG pipeline
  1. Response cache (ใช้ในไฟล์นี้): ถ้าคำถามเหมือนเป๊ะกับที่เคยถามมาก่อน
     ข้ามทั้ง retrieve+rerank+generate ไปเลย ตอบจาก cache ทันที
     -> คุ้มที่สุด เพราะ LLM generation คือขั้นตอนที่แพง/ช้าที่สุดใน pipeline
  2. Embedding cache: ถ้าคำถามซ้ำ ไม่ต้องเรียก embedding API ใหม่
     (ในที่นี้ response cache ครอบคลุมอยู่แล้วเพราะ skip ทั้ง pipeline)

  ใช้ file-based cache (key = hash ของคำถามที่ normalize แล้ว) เพราะเป็น
  local script ขนาดเล็ก ไม่จำเป็นต้องพึ่ง Redis/infra เพิ่ม
  Production จริงที่มี concurrent users ควรย้ายไป Redis เพื่อรองรับ
  หลาย process/instance พร้อมกัน
"""

import os
import json
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from importlib import import_module

rerank_module = import_module("5_rerank")

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

GEN_MODEL = "gpt-4o-mini"
CACHE_DIR = Path("response_cache")
CACHE_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับโรคทุเรียน โดยอ้างอิงจากเอกสารที่ให้มาเท่านั้น

กฎสำคัญ:
1. ตอบโดยอ้างอิงจาก "ข้อมูลอ้างอิง" ที่ให้มาเท่านั้น ห้ามเดาหรือใช้ความรู้ภายนอก
2. ถ้าข้อมูลอ้างอิงไม่มีคำตอบ ให้บอกตรงๆ ว่า "ไม่พบข้อมูลนี้ในเอกสาร" ห้ามแต่งคำตอบขึ้นมาเอง
3. ตอบเป็นภาษาไทย กระชับ ชัดเจน
4. ท้ายคำตอบให้ระบุว่าอ้างอิงจากหน้าไหนบ้าง เช่น (อ้างอิง: หน้า 1, 2)"""


def normalize_question(question: str) -> str:
    return question.strip().lower()


def cache_key(question: str) -> str:
    normalized = normalize_question(question)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def get_cached_response(question: str) -> dict | None:
    path = CACHE_DIR / f"{cache_key(question)}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_cached_response(question: str, result: dict) -> None:
    path = CACHE_DIR / f"{cache_key(question)}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def build_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"(หน้า {c['page']})\n{c['text']}" for c in chunks
    )


def answer_question(question: str, use_cache: bool = True) -> dict:
    if use_cache:
        cached = get_cached_response(question)
        if cached:
            cached["from_cache"] = True
            return cached

    chunks = rerank_module.search_and_rerank(question, retrieve_k=8, final_k=3)
    context = build_context(chunks)

    response = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"ข้อมูลอ้างอิง:\n{context}\n\nคำถาม: {question}"},
        ],
        temperature=0,
    )

    result = {
        "question": question,
        "answer": response.choices[0].message.content,
        "sources": [{"page": c["page"], "chunk_id": c["id"]} for c in chunks],
        "from_cache": False,
    }

    if use_cache:
        save_cached_response(question, result)

    return result


if __name__ == "__main__":
    test_question = "โรครากเน่าโคนเน่ามีอาการอย่างไร"

    print("=== รอบแรก (ไม่มี cache) ===")
    import time
    start = time.time()
    result1 = answer_question(test_question)
    print(f"เวลา: {time.time() - start:.2f}s | from_cache: {result1['from_cache']}")
    print(f"\nคำตอบ: {result1['answer']}")
    print(f"แหล่งอ้างอิง: {result1['sources']}")

    print("\n=== รอบสอง (ถามซ้ำ - ควรมาจาก cache) ===")
    start = time.time()
    result2 = answer_question(test_question)
    print(f"เวลา: {time.time() - start:.2f}s | from_cache: {result2['from_cache']}")
    print(f"\nคำตอบ: {result2['answer']}")
