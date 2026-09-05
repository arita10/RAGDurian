from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

# Load processor and model
processor = TrOCRProcessor.from_pretrained('openthaigpt/thai-trocr')
model = VisionEncoderDecoderModel.from_pretrained('openthaigpt/thai-trocr')

# Load an image
image_path = r"C:\Users\msapi\OneDrive\Documents\RichProject\RAGDurian\Test-THOCR\th.jpg"
image = Image.open(image_path).convert("RGB")

# Process and generate text
pixel_values = processor(images=image, return_tensors="pt").pixel_values
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
