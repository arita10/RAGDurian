"""
STEP 1: Extract text จาก PDF
=============================
ปัญหาที่เจอ: durian.pdf ใช้ font แบบเก่า (TP Kubua) ที่ embed CID mapping
ไม่ตรงกับ Unicode มาตรฐาน -> pdfplumber.extract_text() ตรงๆ จะได้ตัวอักษรขยะ (mojibake)
เช่น '�ä�ͧ�����¹' แทนที่จะเป็น 'โรคทุเรียน'

ทางแก้: แปลงแต่ละหน้า PDF เป็นรูปภาพ แล้วให้ GPT-4o (vision) อ่านและถอดข้อความออกมา
แทนการดึง text layer ตรงๆ วิธีนี้แม่นยำกว่า OCR ทั่วไปมากสำหรับภาษาไทย
เพราะโมเดลเข้าใจบริบทของคำ ไม่ใช่แค่จับรูปทรงตัวอักษร

Output: durian_pages.json -> list ของ {page, text} ที่เป็นภาษาไทยอ่านได้จริง
"""

import os
import re
import json
import base64
from io import BytesIO

import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

PDF_PATH = "durian.pdf"
OUTPUT_PATH = "durian_pages.json"


THAI_BLOCK = re.compile(r"[฀-๿]")
LETTER_LIKE = re.compile(r"[^\W\d_]", re.UNICODE)  # ตัวอักษรทุกภาษา (ไม่รวมตัวเลข/สัญลักษณ์)


def is_garbled_thai(text: str, min_thai_ratio: float = 0.3) -> bool:
    """
    ตรวจสอบว่า text มีปัญหา encoding เพี้ยนหรือไม่

    วิธีที่ 1 (ไม่พอ): นับ replacement character '�' (U+FFFD)
    -> พลาดได้ เพราะฟอนต์ที่ mapping ผิดบางตัวไม่ได้ให้ U+FFFD ตรงๆ
       แต่ mis-map ไปเป็นอักขระ Latin/สัญลักษณ์อื่นที่ "มีอยู่จริง" ใน Unicode
       เช่น 'ä', 'ͧ', '¹' ซึ่งไม่ถูกนับว่าเป็นตัวอักษรเสีย

    วิธีที่ 2 (ใช้จริง): เอกสารนี้ควรเป็นภาษาไทยเกือบทั้งหมด
    ถ้า text ที่ได้มามีสัดส่วนตัวอักษรอยู่ใน Thai Unicode block
    (U+0E00-U+0E7F) ต่ำกว่า threshold เมื่อเทียบกับตัวอักษรทั้งหมด
    แปลว่า font mapping ผิดพลาด ต้องใช้ vision อ่านแทน
    """
    if not text:
        return True

    total_letters = len(LETTER_LIKE.findall(text))
    if total_letters == 0:
        return True

    thai_letters = len(THAI_BLOCK.findall(text))
    ratio = thai_letters / total_letters
    return ratio < min_thai_ratio


def page_to_base64_png(page, resolution: int = 200) -> str:
    """แปลงหน้า PDF เป็นรูปภาพ PNG แล้ว encode เป็น base64 สำหรับส่งให้ vision API"""
    pil_image = page.to_image(resolution=resolution).original
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_page_with_vision(image_b64: str) -> str:
    """ส่งรูปหน้า PDF ให้ GPT-4o ถอดข้อความภาษาไทยออกมาเป็น plain text"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "อ่านข้อความทั้งหมดในรูปภาพนี้ (เอกสารภาษาไทยเกี่ยวกับโรคทุเรียน) "
                            "แล้วถอดออกมาเป็นข้อความล้วน (plain text) ตามลำดับที่ปรากฏในหน้า "
                            "ถ้ามีตาราง ให้แปลงเป็นรูปแบบ markdown table "
                            "ห้ามสรุปย่อ ห้ามเพิ่มความเห็น ถอดคำต่อคำให้ครบถ้วนที่สุด "
                            "ถ้าหน้านี้ไม่มีข้อความ (หน้าว่าง/รูปเปล่า) ให้ตอบว่า EMPTY_PAGE"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def main():
    pages_text = []

    with pdfplumber.open(PDF_PATH) as pdf:
        print(f"เปิด PDF สำเร็จ: {len(pdf.pages)} หน้า")

        for page_number, page in enumerate(pdf.pages, start=1):
            raw_text = page.extract_text() or ""

            if is_garbled_thai(raw_text):
                print(f"หน้า {page_number}: text layer เพี้ยน (encoding ผิด) -> ใช้ Vision อ่านแทน")
                image_b64 = page_to_base64_png(page)
                text = extract_page_with_vision(image_b64)
            else:
                print(f"หน้า {page_number}: text layer ปกติ -> ใช้ text ตรงๆ")
                text = raw_text

            if text and text != "EMPTY_PAGE":
                pages_text.append({"page": page_number, "text": text})
            else:
                print(f"หน้า {page_number}: ไม่มีข้อความ ข้ามหน้านี้")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(pages_text, f, ensure_ascii=False, indent=2)

    print(f"\nสำเร็จ: ดึงข้อความได้ {len(pages_text)} หน้า -> บันทึกที่ {OUTPUT_PATH}")
    print("\n--- ตัวอย่างหน้าแรก ---")
    print(pages_text[0]["text"][:400] if pages_text else "(ไม่มีข้อมูล)")


if __name__ == "__main__":
    main()
