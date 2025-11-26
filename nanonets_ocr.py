import fitz  # PyMuPDF
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import torch
import gc

# ------------------------------
# Load the Nanonets OCR model
# ------------------------------
model_path = "nanonets/Nanonets-OCR2-3B"
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto",           # use GPU if available
    attn_implementation="eager"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

# ------------------------------
# OCR function for a single page
# ------------------------------
def ocr_page_with_nanonets_s(image: Image.Image, model, processor, max_new_tokens=2048):
    prompt = "Extract all text from this image without adding or correcting anything."
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": "image"},  # placeholder, actual image passed separately
            {"type": "text", "text": prompt},
        ]},
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
    
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    output_text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # Free memory
    del inputs, output_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output_text[0]

# ------------------------------
# Process an entire PDF
# ------------------------------
def ocr_pdf_memory_efficient(pdf_path, model, processor):
    doc = fitz.open(pdf_path)
    results = []

    for i, page in enumerate(doc):
        # Render PDF page as image (lower DPI for memory efficiency)
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Downscale large images if needed
        max_width = 1200
        if img.width > max_width:
            new_height = int(img.height * (max_width / img.width))
            img = img.resize((max_width, new_height), Image.LANCZOS)

        # Run OCR on the page
        text = ocr_page_with_nanonets_s(img, model, processor, max_new_tokens=2048)
        results.append(f"--- Page {i + 1} ---\n{text}\n")

        # Free memory
        del img
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return "\n".join(results)

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    pdf_path = "MSTRL-API-684145-003.pdf"
    result_text = ocr_pdf_memory_efficient(pdf_path, model, processor)
    print(result_text)
