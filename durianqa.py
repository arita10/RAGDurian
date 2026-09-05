import pdfplumber
import os
from openai import OpenAI
import chromadb
from sentence_transformers import SentenceTransformer


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

pdf_path = r"C:\Users\msapi\OneDrive\Documents\RichProject\RAGDurian\durian.pdf"  # Replace with your PDF file path
pages_text = []

with pdfplumber.open(pdf_path) as pdf:
    for page_number, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            pages_text.append({
                "page" : page_number ,  # Page numbers are 1-indexed
                "text" : text
            })# Store page number and text

#print(f"ดึงข้อความได้ทั้งหมด {len(pages_text)} หน้า,")
#print(pages_text[0]["text"][:300])

def  split_text(text,chunk_size=500,overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunked_docs = []
for page in pages_text:
    pieces = split_text(page["text"])
    for i,piece in enumerate(pieces):
        chunked_docs.append({
            "id": f'page{page["page"]}_chunk{i}',
            "page": page["page"],
            "text": piece
        })

#print(f"ตัดข้อความได้ทั้งหมด {len(chunked_docs)} ชิ้น")
#print(chunked_docs[0])


embedder = SentenceTransformer('intfloat/multilingual-e5-base')

chroma_client = chromadb.Client()
collection=chroma_client.create_collection(name="durian_diseases")

texts = [d["text"] for d in chunked_docs]
ids = [d["id"] for d in chunked_docs]
metadatas = [{"page": d["page"]} for d in chunked_docs]

embeddings = embedder.encode(texts).tolist()

collection.add(
    ids=ids,
    embeddings = embeddings,
    documents = texts,
    metadatas = metadatas )

print("เก็บข้อมูลลง vector database เรียบร้อย")
print(f"จำนวนชิ้นในฐานข้อมูล: {collection.count()}")

def retrieve (question, top_k=3):
    question_embedding = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k
    )

    retrieve_chunks = []
    for i in range(len(results['ids'][0])):
        retrieve_chunks.append({
  
            "text": results['documents'][0][i],
            "page": results['metadatas'][0][i]['page'],
         
        })
    return retrieve_chunks

test_results = retrieve("โรครากเน่าโคนเน่ามีอาการอย่างไร")

for r in test_results:
    print(f"หน้า {r['page']}: {r['text'][:200]}...")  # Print first 200 characters of each retrieved chunk
    print("---")