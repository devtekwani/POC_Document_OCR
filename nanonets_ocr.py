import fitz  # PyMuPDF
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

# Load your model
model_path = "nanonets/Nanonets-OCR2-3B"
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    dtype="auto",
    device_map="auto",
    attn_implementation="eager"
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_path)
processor = AutoProcessor.from_pretrained(model_path)

# OCR function for a single page (PIL Image)
def ocr_page_with_nanonets_s(image: Image.Image, model, processor, max_new_tokens=4096):
    prompt = """Extract all text from this image without adding or correcting anything."""
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "image", "image": "image"},  # dummy path, actual image passed separately
            {"type": "text", "text": prompt},
        ]},
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)
    
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return output_text[0]

# Function to process PDF and run OCR on each page
def ocr_pdf_with_nanonets(pdf_path, model, processor):
    doc = fitz.open(pdf_path)
    results = []
    
    for page_number in range(len(doc)):
        page = doc[page_number]
        pix = page.get_pixmap(dpi=300)  # convert page to image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text = ocr_page_with_nanonets_s(img, model, processor)
        results.append(f"--- Page {page_number + 1} ---\n{text}\n")
    
    return "\n".join(results)

# Example usage
pdf_path = "MSTRL-API-684145-003.pdf"
result_text = ocr_pdf_with_nanonets(pdf_path, model, processor)
print(result_text)
