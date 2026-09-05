"""
Durian Disease Chatbot - Streamlit UI
=========================================
ห่อ pipeline ที่มีอยู่แล้ว (hybrid search -> rerank -> generate ใน 6_generate.py)
ด้วยหน้าเว็บแชท

Setup ครั้งแรก (ถ้ายังไม่เคยรัน):
  1. python 1_extract_pdf.py
  2. python 2_chunking.py
  3. python 3_embed_and_store.py

รัน: streamlit run streamlit_app.py
"""

import os
from importlib import import_module

import streamlit as st

# บน Streamlit Cloud ไม่มีไฟล์ .env ต้องอ่าน key จาก st.secrets แทน
# (เข้าถึง st.secrets ตรงๆ จะ error ถ้าไม่มีไฟล์ secrets.toml เลย เช่นตอนรัน local ผ่าน .env)
if "OPENAI_API_KEY" not in os.environ:
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

generate_module = import_module("6_generate")

st.set_page_config(page_title="Durian Disease Chatbot", page_icon="🌱")
st.title("🌱 Durian Disease Chatbot")
st.caption("ถามคำถามเกี่ยวกับโรคทุเรียน คำตอบอ้างอิงจากเอกสารเท่านั้น")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("พิมพ์คำถามเกี่ยวกับโรคทุเรียน...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาคำตอบ..."):
            result = generate_module.answer_question(question)

        answer = result["answer"]
        pages = sorted(set(s["page"] for s in result["sources"]))
        cache_note = " _(จาก cache)_" if result["from_cache"] else ""

        content = f"{answer}{cache_note}\n\n**อ้างอิงจากหน้า:** {', '.join(map(str, pages))}"
        st.markdown(content)

    st.session_state.messages.append({"role": "assistant", "content": content})
