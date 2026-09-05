"""
STEP 2: แบ่ง Chunk + Overlap
==============================
ทำไมต้อง chunk?
  Embedding model บีบอัดข้อความทั้งก้อนให้เป็น vector เดียว ถ้าข้อความยาวเกินไป
  (เช่น embed ทั้งหน้า) vector จะ "เบลอ" เพราะพยายามแทนหลายประเด็นพร้อมกัน
  พอ user ถามคำถามเจาะจง similarity search จะแม่นน้อยลง
  -> ต้องตัดเอกสารเป็นชิ้นเล็กๆ ที่แต่ละชิ้น "พูดเรื่องเดียวชัดๆ"

ทำไมต้อง overlap?
  ถ้าตัด chunk แบบชนกันพอดี (ไม่ overlap) มีความเสี่ยงที่ประโยคสำคัญจะถูก
  "ตัดขาดครึ่ง" อยู่ตรงรอยต่อพอดี ทำให้ทั้งสอง chunk ข้างเคียงต่างก็ไม่มี
  บริบทที่สมบูรณ์ -> ใส่ overlap (เอาท้าย chunk ก่อนหน้ามาซ้อนต้น chunk ถัดไป)
  เพื่อประกันว่าอย่างน้อยจะมี chunk หนึ่งที่มีประโยคนั้นครบถ้วน

กลยุทธ์ที่ใช้: Hierarchical / Recursive splitting
  เอกสารนี้มีโครงสร้างชัดเจน (สาเหตุ / ลักษณะอาการ / การแพร่ระบาด / การป้องกันกำจัด)
  จึงตัดตามลำดับความสำคัญของ separator:
    1. หัวข้อย่อยของโรค (สาเหตุ, ลักษณะอาการ, ฯลฯ)  <- ตัดตรงนี้ก่อนเสมอถ้าเจอ
    2. ย่อหน้า/บรรทัด (\n)
    3. ประโยค (จุด, ฯลฯ)
    4. ตัวอักษรตรงๆ (fallback สุดท้ายถ้ายังยาวเกิน)
  ตรงข้ามกับโค้ดเดิมที่ตัดตามจำนวนตัวอักษรตรงๆ โดยไม่สนใจโครงสร้างเลย

Output: durian_chunks.json
"""

import json
import re

INPUT_PATH = "durian_pages.json"
OUTPUT_PATH = "durian_chunks.json"

CHUNK_SIZE = 400      # ขนาด chunk เป้าหมาย (ตัวอักษร) - พอดีกับ 1 หัวข้อย่อยส่วนใหญ่
CHUNK_OVERLAP = 80    # overlap ~20% ของ chunk size
MIN_CHUNK_SIZE = 40   # chunk ที่สั้นกว่านี้ (เช่น แค่หัวข้อลอยๆ) จะถูก merge เข้ากับชิ้นถัดไป

# หัวข้อย่อยที่พบซ้ำๆ ในเอกสารนี้ ใช้เป็นจุดตัดหลัก (priority สูงสุด)
SECTION_HEADERS = [
    "สาเหต",
    "ลักษณะอาการ",
    "การแพร่ระบาด",
    "การป้องกันกำจัด",
]


def split_by_section_headers(text: str) -> list[str]:
    """ตัด text ตรงจุดที่เจอหัวข้อย่อย (สาเหตุ/ลักษณะอาการ/ฯลฯ) โดยหัวข้อนั้นติดไปกับ chunk ถัดไป"""
    pattern = "(" + "|".join(SECTION_HEADERS) + ")"
    parts = re.split(pattern, text)

    sections = []
    buffer = ""
    for part in parts:
        if part in SECTION_HEADERS:
            if buffer.strip():
                sections.append(buffer.strip())
            buffer = part
        else:
            buffer += part
    if buffer.strip():
        sections.append(buffer.strip())
    return sections


def split_by_sentences(text: str, max_len: int) -> list[str]:
    """ถ้า section ยังยาวเกิน max_len ให้ตัดตามบรรทัด/ประโยคต่อ แล้วมัดรวมจนใกล้ max_len"""
    # ตัดตามบรรทัด (\n) ก่อน เพราะเอกสารนี้ตัดขึ้นบรรทัดใหม่ตามประโยค/ข้อย่อยอยู่แล้ว
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    pieces = []
    current = ""
    for line in lines:
        candidate = (current + " " + line).strip() if current else line
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
    if len(pieces) <= 1:
        return pieces

    overlapped = [pieces[0]]
    for i in range(1, len(pieces)):
        prev_tail = pieces[i - 1][-overlap:]
        overlapped.append(prev_tail + " " + pieces[i])
    return overlapped


def merge_short_pieces(pieces: list[str], min_len: int) -> list[str]:
    """merge ชิ้นที่สั้นเกินไป (เช่น หัวข้อลอยๆ ก่อนเจอ 'สาเหตุ') เข้ากับชิ้นถัดไป"""
    merged = []
    buffer = ""
    for piece in pieces:
        buffer = (buffer + " " + piece).strip() if buffer else piece
        if len(buffer) >= min_len:
            merged.append(buffer)
            buffer = ""
    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer
        else:
            merged.append(buffer)
    return merged


def chunk_page_text(text: str) -> list[str]:
    sections = split_by_section_headers(text)

    raw_pieces = []
    for section in sections:
        if len(section) <= CHUNK_SIZE:
            raw_pieces.append(section)
        else:
            raw_pieces.extend(split_by_sentences(section, CHUNK_SIZE))

    raw_pieces = merge_short_pieces(raw_pieces, MIN_CHUNK_SIZE)
    return add_overlap(raw_pieces, CHUNK_OVERLAP)


def main():
    with open(INPUT_PATH, encoding="utf-8") as f:
        pages = json.load(f)

    chunked_docs = []
    for page in pages:
        pieces = chunk_page_text(page["text"])
        for i, piece in enumerate(pieces):
            chunked_docs.append({
                "id": f'page{page["page"]}_chunk{i}',
                "page": page["page"],
                "text": piece,
            })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunked_docs, f, ensure_ascii=False, indent=2)

    lengths = [len(c["text"]) for c in chunked_docs]
    print(f"ตัดข้อความได้ {len(chunked_docs)} chunks จาก {len(pages)} หน้า")
    print(f"ความยาวเฉลี่ย: {sum(lengths)/len(lengths):.0f} ตัวอักษร (min={min(lengths)}, max={max(lengths)})")
    print(f"บันทึกที่ {OUTPUT_PATH}\n")

    print("--- ตัวอย่าง 3 chunks แรก ---")
    for c in chunked_docs[:3]:
        print(f"[{c['id']}] ({len(c['text'])} chars)")
        print(c["text"][:150] + "...")
        print()


if __name__ == "__main__":
    main()
