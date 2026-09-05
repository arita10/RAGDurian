"""
STEP 7: Evaluation (วัดผล RAG pipeline)
============================================
ทำไมต้อง evaluate?
  RAG มี 2 ส่วนที่พังแยกกันได้:
    - Retrieval พัง: ดึง chunk ผิด/ไม่เกี่ยวข้อง -> ต่อให้ LLM เก่งแค่ไหนก็ตอบถูกไม่ได้
    - Generation พัง: context ถูกแล้ว แต่ LLM ตอบผิดหรือ "hallucinate"
      (สร้างข้อมูลที่ไม่มีอยู่ใน context ขึ้นมาเอง)
  ถ้าไม่วัดแยกกัน จะไม่รู้ว่าควรไปแก้ตรงไหน (เช่น ปรับ chunking/embedding
  vs ปรับ prompt/model)

Metrics ที่วัด:
  1. Retrieval:
     - Hit Rate @ K: ในคำถามทดสอบ มีกี่ % ที่หน้าที่ถูกต้อง (ground truth)
       ติดอยู่ใน top-K ที่ระบบดึงมาได้จริง
     - MRR (Mean Reciprocal Rank): อันดับของหน้าที่ถูกต้องอยู่ตำแหน่งไหนโดยเฉลี่ย
       (1/rank) ยิ่งเข้าใกล้ 1.0 ยิ่งดี (แปลว่าหน้าที่ถูกมักติดอันดับ 1 เสมอ)
  2. Generation (ใช้ LLM-as-judge เพราะเป็น metric เชิงคุณภาพ วัดด้วยตัวเลขตรงๆ ยาก):
     - Faithfulness: คำตอบอิงจาก context จริงไหม หรือมีส่วนที่ hallucinate
     - Answer Relevance: คำตอบตรงกับคำถามที่ถามไหม

Test set: สร้างคำถาม + หน้าอ้างอิงที่ถูกต้อง (ground truth) เองจากเนื้อหาเอกสาร
  ในงานจริงควรให้ domain expert (เช่น นักวิชาการเกษตร) ช่วยตรวจสอบ/สร้าง test set นี้
"""

import os
import json

from dotenv import load_dotenv
from openai import OpenAI

from importlib import import_module

hybrid_search_module = import_module("4_hybrid_search")
rerank_module = import_module("5_rerank")
generate_module = import_module("6_generate")

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

JUDGE_MODEL = "gpt-4o-mini"

# Test set: คำถาม + หน้าที่ควรเป็นคำตอบ (ground truth อ้างอิงจาก durian_pages.json)
TEST_SET = [
    {"question": "โรครากเน่าโคนเน่ามีอาการอย่างไร", "expected_pages": [1]},
    {"question": "โรคผลเน่าเกิดจากเชื้ออะไร", "expected_pages": [2]},
    {"question": "โรคใบติดหรือใบไหม้มีสาเหตุจากเชื้อชนิดใด", "expected_pages": [4]},
    {"question": "โรคราแป้งป้องกันกำจัดอย่างไร", "expected_pages": [8]},
    {"question": "ทำไมดอกทุเรียนถึงร่วง", "expected_pages": [10]},
    {"question": "โรคราสีชมพูมีลักษณะอาการอย่างไร", "expected_pages": [7]},
]


def evaluate_retrieval(test_set: list[dict], top_k: int = 5) -> dict:
    """วัด Hit Rate@K และ MRR ของ hybrid search (ก่อน rerank)"""
    hits = 0
    reciprocal_ranks = []

    for case in test_set:
        results = hybrid_search_module.hybrid_search(case["question"], top_k=top_k)
        retrieved_pages = [r["page"] for r in results]

        hit = any(p in retrieved_pages for p in case["expected_pages"])
        hits += int(hit)

        rank = None
        for i, p in enumerate(retrieved_pages, start=1):
            if p in case["expected_pages"]:
                rank = i
                break
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

    return {
        "hit_rate": hits / len(test_set),
        "mrr": sum(reciprocal_ranks) / len(test_set),
        "n_cases": len(test_set),
    }


def judge_faithfulness(question: str, context: str, answer: str) -> dict:
    """LLM-as-judge: คำตอบอิงจาก context จริงไหม (ไม่ hallucinate)"""
    prompt = f"""ตรวจสอบว่าคำตอบต่อไปนี้ "อิงจากข้อมูลอ้างอิง" ที่ให้มาจริงหรือไม่
หรือมีส่วนที่ผู้ตอบสร้างขึ้นเอง (hallucinate) ที่ไม่มีอยู่ในข้อมูลอ้างอิง

ข้อมูลอ้างอิง:
{context}

คำถาม: {question}
คำตอบที่ต้องตรวจสอบ: {answer}

ให้คะแนน faithfulness 0-10 (10 = ทุกประโยคในคำตอบมีหลักฐานรองรับใน context ครบถ้วน,
0 = คำตอบส่วนใหญ่ไม่มีอยู่ใน context เลย)
ตอบเป็น JSON: {{"score": <0-10>, "reason": "<เหตุผลสั้นๆ>"}}"""

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def judge_relevance(question: str, answer: str) -> dict:
    """LLM-as-judge: คำตอบตรงกับคำถามที่ถามไหม"""
    prompt = f"""คำถาม: {question}
คำตอบ: {answer}

คำตอบนี้ตอบคำถามที่ถามตรงประเด็นแค่ไหน? ให้คะแนน 0-10
(10 = ตอบตรงประเด็นครบถ้วน, 0 = ไม่ตอบคำถามที่ถามเลย)
ตอบเป็น JSON: {{"score": <0-10>, "reason": "<เหตุผลสั้นๆ>"}}"""

    response = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def evaluate_generation(test_set: list[dict]) -> dict:
    """วัด faithfulness + relevance ของคำตอบสุดท้าย (ผ่าน pipeline เต็ม)"""
    faithfulness_scores = []
    relevance_scores = []
    details = []

    for case in test_set:
        chunks = rerank_module.search_and_rerank(case["question"], retrieve_k=8, final_k=3)
        context = generate_module.build_context(chunks)
        result = generate_module.answer_question(case["question"], use_cache=False)

        faith = judge_faithfulness(case["question"], context, result["answer"])
        rel = judge_relevance(case["question"], result["answer"])

        faithfulness_scores.append(faith["score"])
        relevance_scores.append(rel["score"])
        details.append({
            "question": case["question"],
            "faithfulness": faith,
            "relevance": rel,
        })

    return {
        "avg_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
        "avg_relevance": sum(relevance_scores) / len(relevance_scores),
        "details": details,
    }


if __name__ == "__main__":
    print("=== วัดผล Retrieval (Hit Rate @5, MRR) ===")
    retrieval_results = evaluate_retrieval(TEST_SET, top_k=5)
    print(f"Hit Rate@5: {retrieval_results['hit_rate']:.2%}")
    print(f"MRR: {retrieval_results['mrr']:.3f}")
    print(f"จำนวนคำถามทดสอบ: {retrieval_results['n_cases']}")

    print("\n=== วัดผล Generation (Faithfulness, Relevance) ===")
    gen_results = evaluate_generation(TEST_SET)
    print(f"เฉลี่ย Faithfulness: {gen_results['avg_faithfulness']:.1f}/10")
    print(f"เฉลี่ย Relevance: {gen_results['avg_relevance']:.1f}/10")

    print("\n--- รายละเอียดแต่ละคำถาม ---")
    for d in gen_results["details"]:
        print(f"Q: {d['question']}")
        print(f"  Faithfulness: {d['faithfulness']['score']}/10 - {d['faithfulness']['reason']}")
        print(f"  Relevance: {d['relevance']['score']}/10 - {d['relevance']['reason']}")
