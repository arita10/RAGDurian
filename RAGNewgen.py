import sys
# ตั้งค่า stdout/stderr ให้บังคับแสดงผล UTF-8 บน terminal เสมอ
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import pymupdf  # ใช้ pymupdf แทน fitz เพื่อแก้ deprecation warning
import re


import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter


import os
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings


def extract_clean_text_from_pdf(pdf_path: str) -> list[dict]:
    # เปิดไฟล์ด้วย pymupdf
    doc = pymupdf.open(pdf_path)
    pages_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # ดึงข้อความ
        raw_text = page.get_text("text")

        # ทำความสะอาดข้อความพื้นฐาน
        cleaned = re.sub(r'(\w+)-\n(\w+)', r'\1\2', raw_text)
        cleaned = re.sub(r'\n\s*\d+\s*\n', '\n', cleaned)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        cleaned = cleaned.strip()

        if cleaned:
            pages_data.append({
                "page": page_num + 1,
                "text": cleaned
            })

    return pages_data

# รันและทดสอบปริ้นท์
extracted_pages = extract_clean_text_from_pdf(r"C:\Users\msapi\OneDrive\Documents\RichProject\RAGDurian\durian.pdf")
print(f"สกัดเสร็จสิ้น: ทั้งหมด {len(extracted_pages)} หน้า")

# ลองปริ้นท์ตัวอย่างหน้าแรกออกมาดูเนื้อหาภาษาไทย
if extracted_pages:
    print("\n--- ตัวอย่างเนื้อหาหน้า 1 ---")
    print(extracted_pages[0]["text"][:300])




# 1. กำหนดตัวแบ่งก้อนใหญ่ (Parent) - เก็บใจความครบถ้วน
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", " ", ""]
)

# 2. กำหนดตัวแบ่งก้อนย่อย (Child) - สำหรับ Vector Search คมๆ
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=40,
    separators=["\n\n", "\n", " ", ""]
)

processed_chunks = []

# นำ extracted_pages จาก Step 1 มาประมวลผล
for page in extracted_pages:
    # ตัดเป็น Parent Chunks ก่อน
    parent_docs = parent_splitter.create_documents(
        texts=[page["text"]],
        metadatas=[{"page": page["page"]}]
    )

    for p_doc in parent_docs:
        # สร้าง Parent ID กำกับไว้
        parent_id = str(uuid.uuid4())
        
        # ตัด Parent ก้อนนี้ออกเป็น Child Chunks หลายๆ ชิ้น
        child_docs = child_splitter.create_documents(
            texts=[p_doc.page_content],
            metadatas=[{
                "page": page["page"],
                "parent_id": parent_id,
                "parent_text": p_doc.page_content  # ฝากเนื้อหาตัวเต็มของ Parent ไว้ใน metadata
            }]
        )
        processed_chunks.extend(child_docs)

print(f"แบ่งชิ้นส่วนเรียบร้อย:")
print(f"- จำนวน Child Chunks ทั้งหมด: {len(processed_chunks)} ชิ้น")

# ลองปริ้นท์ดูตัวอย่าง Child Chunk ก้อนแรก
if processed_chunks:
    sample = processed_chunks[0]
    print("\n--- ตัวอย่าง Child Chunk (ใช้ทำ Vector Search) ---")
    print(sample.page_content)
    print("\n--- Parent Chunk ที่ผูกติดกัน (ใช้ส่งให้ LLM อ่าน) ---")
    print(sample.metadata["parent_text"][:300] + "...")


# ตรวจสอบว่าตั้งค่า OPENAI_API_KEY เรียบร้อยแล้ว
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# 1. ตั้งค่า Embedding Model (สำหรับภาษาไทย text-embedding-3-small ถือว่าคุ้มค่าและเก่งมาก)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("กำลังสร้าง Vector Database และ Index...")

# 2. Dense Vector Store (Chroma)
# นำเฉพาะ child_chunks ที่เราหั่นไว้ใน Step 2 มาสร้าง Index
vectorstore = Chroma.from_documents(
    documents=processed_chunks,
    embedding=embeddings,
    collection_name="pdf_rag_child_chunks"
)
# ดึงชิ้นส่วนที่ใกล้เคียงที่สุด 8 ชิ้นแรก
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

# 3. Sparse Keyword Search (BM25)
# คำนวณ BM25 จาก Child Chunks ชุดเดียวกัน
bm25_retriever = BM25Retriever.from_documents(documents=processed_chunks)
bm25_retriever.k = 8

# 4. รวมพลังเป็น Hybrid Retriever (Ensemble)
# กำหนด weight: Dense 60% เพื่อจับความหมาย + BM25 40% ดักคีย์เวิร์ดตรงตัว
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6]
)

print("สร้าง Hybrid Retriever สำเร็จ!")