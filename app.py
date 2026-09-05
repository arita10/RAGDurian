"""
APP: รวม RAG pipeline ทั้งหมดเป็น chatbot ใช้งานจริง
========================================================
Pipeline เต็ม: Hybrid Search -> Rerank -> Generate (with caching)

Setup ครั้งแรก (รันทีละไฟล์ตามลำดับ ก่อนมาใช้ app.py นี้):
  1. python 1_extract_pdf.py       -> durian_pages.json
  2. python 2_chunking.py          -> durian_chunks.json
  3. python 3_embed_and_store.py   -> chroma_db/, bm25_index.pkl

จากนั้นรัน: python app.py
"""

from importlib import import_module

generate_module = import_module("6_generate")


def chat():
    print("=" * 50)
    print("Durian Disease Chatbot (พิมพ์ 'exit' เพื่อออก)")
    print("=" * 50)

    while True:
        question = input("\nคำถาม: ").strip()
        if question.lower() in ("exit", "quit", "ออก"):
            break
        if not question:
            continue

        result = generate_module.answer_question(question)

        cache_note = " (จาก cache)" if result["from_cache"] else ""
        print(f"\nคำตอบ{cache_note}:\n{result['answer']}")
        pages = sorted(set(s["page"] for s in result["sources"]))
        print(f"\n[อ้างอิงจากหน้า: {', '.join(map(str, pages))}]")


if __name__ == "__main__":
    chat()
