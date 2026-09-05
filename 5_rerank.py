"""
STEP 5: Reranking
=====================
ทำไม hybrid search ยังไม่พอ?
  จาก step 4 จะเห็นว่าคำถาม "โรครากเน่าโคนเน่ามีอาการอย่างไร" ได้ chunk ที่พูดถึง
  "การป้องกันกำจัด" และ "โรคผลเน่า" ติดอันดับต้นๆ แทนที่จะเป็น chunk ที่อธิบาย
  "อาการ" ตรงๆ เพราะ embedding + BM25 มองแค่ "ความเกี่ยวข้องเชิงคำ/ความหมายกว้างๆ"
  ไม่ได้เข้าใจว่า user ถามเจาะจงเรื่อง "อาการ" ไม่ใช่ "การป้องกัน"

Bi-encoder vs Cross-encoder:
  - Bi-encoder (ที่ใช้ตอน embed ใน step 3): encode query กับ document แยกกัน
    เป็น vector คนละตัว แล้วเทียบ similarity ทีหลัง -> เร็ว (embed ล่วงหน้าได้)
    แต่โมเดลไม่เคย "เห็น" query กับ document คู่กันตอน encode เสียความแม่นยำ
  - Cross-encoder (reranker): เอา query + document ใส่โมเดลพร้อมกันในครั้งเดียว
    ให้ดูทั้งคู่แล้วให้คะแนนตรงๆ -> แม่นกว่ามาก แต่ช้า (คำนวณทีละคู่)
    ทำกับเอกสารทั้งหมดไม่ได้ ต้องทำแค่กับ candidates ที่คัดมาแล้ว

Pattern มาตรฐาน: retrieve กว้างๆ ก่อน (เร็ว) -> แล้วค่อย rerank เฉพาะ candidates
  (แม่น แต่ทำแค่ไม่กี่ชิ้นเพราะช้า)

ในที่นี้ใช้ LLM-based reranking (ให้ GPT ให้คะแนนความเกี่ยวข้องของแต่ละ chunk
กับคำถามโดยตรง) แทนการติดตั้ง cross-encoder model แยก เพราะ:
  - ยืดหยุ่นกว่า ปรับ prompt ได้ตามงาน
  - แม่นยำสำหรับภาษาไทยมากกว่า cross-encoder สำเร็จรูปทั่วไปที่ train ด้วยอังกฤษเป็นหลัก
  - ไม่ต้องติดตั้ง/โหลดโมเดลเพิ่มในเครื่อง
"""

import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from importlib import import_module

hybrid_search_module = import_module("4_hybrid_search")

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

RERANK_MODEL = "gpt-4o-mini"  # โมเดลเล็ก พอสำหรับงานให้คะแนนความเกี่ยวข้อง ไม่ต้องใช้โมเดลใหญ่


def rerank(question: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """
    ให้ LLM ให้คะแนนความเกี่ยวข้อง (0-10) ของแต่ละ chunk เทียบกับคำถาม
    แล้วเรียงใหม่ตามคะแนนนั้น คืนค่า top_k อันดับแรก
    """
    numbered_chunks = "\n\n".join(
        f"[{i}] {c['text']}" for i, c in enumerate(candidates)
    )

    prompt = f"""คำถามของผู้ใช้: {question}

ต่อไปนี้คือข้อความที่ดึงมาได้ (แต่ละอันมีหมายเลขกำกับ):

{numbered_chunks}

ให้คะแนนความเกี่ยวข้องของแต่ละข้อความกับคำถาม (0 = ไม่เกี่ยวเลย, 10 = ตอบคำถามได้ตรงที่สุด)
พิจารณาว่าข้อความนั้น "ตอบคำถามที่ถามจริงๆ" หรือแค่พูดเรื่องใกล้เคียงกัน (เช่น
ถ้าถามเรื่อง "อาการ" แต่ข้อความพูดเรื่อง "การป้องกัน" ควรได้คะแนนต่ำกว่า
ข้อความที่บรรยายอาการตรงๆ)

ตอบเป็น JSON array เท่านั้น รูปแบบ: [{{"index": 0, "score": 8}}, {{"index": 1, "score": 3}}, ...]
ต้องมีครบทุก index ที่ให้มา"""

    response = client.chat.completions.create(
        model=RERANK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )

    # ขอ JSON object ห่อ array ไว้ เพราะ response_format=json_object ต้องการ root เป็น object
    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
        scores = parsed if isinstance(parsed, list) else parsed.get("results", parsed.get("scores", []))
    except (json.JSONDecodeError, AttributeError):
        # fallback: ถ้า parse ไม่ได้ ให้คืนลำดับเดิมจาก hybrid search
        return candidates[:top_k]

    score_map = {item["index"]: item["score"] for item in scores}
    ranked = sorted(
        range(len(candidates)),
        key=lambda i: score_map.get(i, 0),
        reverse=True,
    )

    reranked = []
    for i in ranked[:top_k]:
        chunk = dict(candidates[i])
        chunk["rerank_score"] = score_map.get(i, 0)
        reranked.append(chunk)
    return reranked


def search_and_rerank(question: str, retrieve_k: int = 8, final_k: int = 3) -> list[dict]:
    """pipeline เต็ม: hybrid search (กว้าง) -> rerank (แคบลง แม่นขึ้น)"""
    candidates = hybrid_search_module.hybrid_search(question, top_k=retrieve_k)
    return rerank(question, candidates, top_k=final_k)


if __name__ == "__main__":
    test_question = "โรครากเน่าโคนเน่ามีอาการอย่างไร"
    print(f"คำถาม: {test_question}\n")

    print("=== ก่อน rerank (hybrid search ตรงๆ) ===")
    before = hybrid_search_module.hybrid_search(test_question, top_k=5)
    for r in before:
        print(f"[{r['id']}] page={r['page']}: {r['text'][:100]}...")

    print("\n=== หลัง rerank ===")
    after = search_and_rerank(test_question, retrieve_k=8, final_k=3)
    for r in after:
        print(f"[{r['id']}] page={r['page']} rerank_score={r['rerank_score']}: {r['text'][:100]}...")
