import fitz  # PyMuPDF
from transformers import AutoProcessor
from transformers import HunYuanVLForConditionalGeneration
from PIL import Image
import torch
import os


def clean_repeated_substrings(text):
    """Clean repeated substrings in text"""
    print("Cleaning repeated substrings...")
    n = len(text)
    if n < 8000:
        return text
    for length in range(2, n // 10 + 1):
        candidate = text[-length:]
        count = 0
        i = n - length

        while i >= 0 and text[i:i + length] == candidate:
            count += 1
            i -= length

        if count >= 10:
            return text[:n - length * (count - 1)]
    return text


# ------------------------------
# Load Model and Processor
# ------------------------------
model_name_or_path = "tencent/HunyuanOCR"

processor = AutoProcessor.from_pretrained(model_name_or_path, use_fast=False)

model = HunYuanVLForConditionalGeneration.from_pretrained(
    model_name_or_path,
    attn_implementation="eager",
    dtype=torch.bfloat16,
    device_map="auto"
)

device = next(model.parameters()).device


# ------------------------------
# OCR Function for a Single Image
# ------------------------------
def ocr_image(image_path):
    print(f"Running OCR on image: {image_path}")
    image = Image.open(image_path)

    messages = [
        {"role": "system", "content": ""},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": "Extract the text from the above document as if you were reading it naturally. Return the tables in html format. Return the equations in LaTeX representation. If there is an image in the document and image caption is not present, add a small description of the image inside the <img></img> tag; otherwise, add the image caption inside <img></img>. Watermarks should be wrapped in brackets. Ex: <watermark>OFFICIAL COPY</watermark>. Page numbers should be wrapped in brackets. Ex: <page_number>14</page_number> or <page_number>9/22</page_number>. Prefer using ☐ and ☑ for check boxes"}
            ]
        }]



    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("\nText Prompt:", text_prompt )

    inputs = processor(
        text=[text_prompt],
        images=image,
        padding=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=16384,
            do_sample=False
        )

    # Trim input prompt tokens
    in_ids = inputs.input_ids
    gen_ids = generated_ids[:, in_ids.shape[1]:]

    output_text = processor.batch_decode(
        gen_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    output_text = clean_repeated_substrings(output_text)
    print("OCR Result for path:", image_path , "\n", output_text)
    return output_text


# ------------------------------
# PDF → images → OCR
# ------------------------------
def ocr_pdf(pdf_path, dpi=150):
    doc = fitz.open(pdf_path)
    results = []

    os.makedirs("pdf_pages", exist_ok=True)

    for i, page in enumerate(doc):
        print(f"Processing page {i+1}/{len(doc)} ...")

        # Convert to image
        pix = page.get_pixmap(dpi=dpi)
        img_path = f"pdf_pages/page_{i+1}.png"
        pix.save(img_path)

        # Run OCR
        text = ocr_image(img_path)
        results.append(text)

    doc.close()
    return results


# ------------------------------
# Run OCR on a PDF
# ------------------------------
pdf_path = "MSTRL-API-684145-003.pdf"

all_text = ocr_pdf(pdf_path)

print("\n===== OCR RESULT =====\n")
for i, page_text in enumerate(all_text, 1):
    print(f"\n--- PAGE {i} ---\n")
    print(page_text)
