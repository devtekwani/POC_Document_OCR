from transformers import AutoModel, AutoTokenizer
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_id = 'Jalea96/DeepSeek-OCR-bnb-4bit-NF4'

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_id, 
    _attn_implementation='eager',
    trust_remote_code=True, 
    use_safetensors=True,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model = model.eval()

# --- 1. Set Image and Task Prompt ---
prompt = "Convert the document to markdown. "
image_file = 'img2.png'
output_path = 'output'

if not os.path.exists(output_path):
    os.makedirs(output_path)

# --- 2. Set Resolution ---
# (Gundam is recommended for most documents)
# Tiny:  base_size = 512,  image_size = 512, crop_mode = False
# Small: base_size = 640,  image_size = 640, crop_mode = False
# Base:  base_size = 1024, image_size = 1024, crop_mode = False
# Large: base_size = 1280, image_size = 1280, crop_mode = False
# Gundam:base_size = 1024, image_size = 640, crop_mode = True
base_size, image_size, crop_mode = 1024, 640, True

# --- 3. Run Inference ---
res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path=output_path, 
    base_size=base_size, 
    image_size=image_size, 
    crop_mode=crop_mode, 
    save_results=True,
    test_compress=True
)
