import os                          # ใช้อ่าน environment variable เช่น API key
import re                          # ใช้ regular expression ตรวจจับ/ตัดข้อความ
from dotenv import load_dotenv     # โหลดค่าจากไฟล์ .env เข้าเป็น environment variable
from openai import OpenAI          # SDK เรียกใช้ OpenAI (embedding + LLM)
import pdfplumber                  # อ่านไฟล์ PDF แล้วดึงข้อความ/รูปภาพออกมา

import base64
from io import BytesIO

load_dotenv()                                              # อ่านไฟล์ .env แล้วตั้งเป็น environment variable ให้อัตโนมัติ
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])      # สร้าง client เดียวไว้เรียก OpenAI API ทั้งไฟล์ (embedding + chat)

THAI_BLOCK = re.compile(r"[฀-๿]")
LETTER_LIKE = re.compile(r"[^\W\d_]", re.UNICODE)

def is_garbled_thai(text: str,min_thai_ratio: float = 0.3) -> bool:
    """ตรวจสอบว่า text เป็นภาษาไทยหรือไม่ โดยดูสัดส่วนตัวอักษรไทยต่อจำนวนตัวอักษรทั้งหมด"""
    if not text:
        return True
    
    total_letters = len(LETTER_LIKE.findall(text))
    if total_letters == 0:
        return True

    thai_letters = len(THAI_BLOCK.findall(text))
    ratio = thai_letters / total_letters
    return ratio < min_thai_ratio

def page_to_base64_png(page, resolution: int = 200) -> str:
    """แปลงหน้า PDF เป็น base64 PNG"""
    image = page.to_image(resolution=resolution).original
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def extract_page_with_vision(image_b64: str) -> str:
    """ส่ง base64 PNG ไปให้ OpenAI Vision อ่านข้อความออกมา"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
             {
                "role": "user",                              # ข้อความนี้มาจากฝั่ง "ผู้ใช้" (เทียบกับ system/assistant)
                "content": [                                 # content เป็น list ได้ เพราะมีทั้ง text และ image ในข้อความเดียว
                    {
                        "type": "text",                       # ส่วนแรกคือคำสั่ง (instruction) เป็นข้อความ
                        "text": (
                            "อ่านข้อความทั้งหมดในรูปภาพนี้ (เอกสารภาษาไทยเกี่ยวกับโรคทุเรียน) "
                            "แล้วถอดออกมาเป็นข้อความล้วน (plain text) ตามลำดับที่ปรากฏในหน้า "
                            "ถ้ามีตาราง ให้แปลงเป็นรูปแบบ markdown table "
                            "ห้ามสรุปย่อ ห้ามเพิ่มความเห็น ถอดคำต่อคำให้ครบถ้วนที่สุด "
                            "ถ้าหน้านี้ไม่มีข้อความ (หน้าว่าง/รูปเปล่า) ให้ตอบว่า EMPTY_PAGE"
                        ),
                    },
                    {
                        "type": "image_url",                  # ส่วนที่สองคือตัวรูปภาพ
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},  # ฝังรูปเป็น data URI ตามที่อธิบายไปก่อนหน้า
                    },
                ],
            }
        ],
        temperature=0.0,  # ลดความสุ่มของผลลัพธ์ (0.0 = ตรงไปตรงมา, 1.0 = สุ่มมาก)
    )
    return response.choices[0].message.content.strip()  # ด

def extract_pdf_text(pdf_path: str) -> list[str]:
    """อ่านไฟล์ PDF แล้วดึงข้อความออกมาเป็น list ของแต่ละหน้า"""
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number,page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""

            if is_garbled_thai(raw_text):
                # ถ้าข้อความในหน้านี้เป็นภาษาไทยที่อ่านไม่ออก ให้ใช้ OpenAI Vision อ่านแทน
                image_b64 = page_to_base64_png(page)
                text = extract_page_with_vision(image_b64)
            else:
                text = raw_text

            if text and text != "EMPTY_PAGE":
                pages_text.append({"page":page_number, "text": text})
            
    return pages_text

SECTION_HEADERS = ["สาเหต", "ลักษณะอาการ", "การแพร่ระบาด", "การป้องกันกำจัด"]

def split_by_section_headers(text: str)->list[str]:
    pattern = "(" + "|".join(SECTION_HEADERS) + ")"
    parts = re.split(pattern, text)

    sections = []
    buffer = ""
    for part in parts:
        if part in SECTION_HEADERS:
            if buffer:
                sections.append(buffer.strip())
               
            buffer = part
        else:
            buffer += part
    if buffer.strip():
        sections.append(buffer.strip())
    return sections

def split_by_sentences(text: str, max_len:int) -> list[str]:
    lines =[l.strip() for l in text.spli]("\n") if l.strip()]

    pieces = []
    current = ""
    for line in lines:
        candidate = (current + ""+line).strip() 
        if len(candidate) > max_len and current:
            pieces.append(current)
            current = line
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces

def add_overlap(pieces: list[str], overlap: int) -> list[str]:
    """เติม overlap โดยเอาท้ายของชิ้นก่อนหน้ามาต่อหน้าชิ้นถัดไป"""
    if len(pieces) <= 1:                                # ถ้ามีแค่ 0-1 ชิ้น ไม่มี "ชิ้นก่อนหน้า" ให้ overlap ได้ -> คืนค่าเดิมกลับไปเลย
        return pieces

    overlapped = [pieces[0]]                             # ชิ้นแรกไม่มีชิ้นก่อนหน้าให้ overlap -> ใส่ไปตามเดิม เป็นจุดเริ่มต้นของ list ใหม่
    for i in range(1, len(pieces)):                       # วนตั้งแต่ชิ้นที่ 2 เป็นต้นไป (index 1)
        prev_tail = pieces[i - 1][-overlap:]               # หยิบ "ท้าย" ของชิ้นก่อนหน้า ยาว `overlap` ตัวอักษร (slice ลบ = นับจากท้าย)
        overlapped.append(prev_tail + " " + pieces[i])     # เอาส่วนท้ายนั้นไปแปะไว้หน้าชิ้นปัจจุบัน แล้วเก็บเป็นชิ้นใหม่
    return overlapped                                       # คืนค่า list ที่มี overlap ครบแล้ว
