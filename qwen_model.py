from transformers import AutoModelForCausalLM, AutoTokenizer

# Load once globally (much faster than loading inside the function)
model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

def generate_response(prompt: str, max_new_tokens: int = 512) -> str:
    """
    Generate a response from Qwen2.5-7B-Instruct given a user prompt.
    """

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    # Prepare chat input text
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate output
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )

    # Remove input tokens from output
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode
    response = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return response

iscontinue = True

while iscontinue:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        iscontinue = False
        print("Exiting chat.")
    else:
        response = generate_response(user_input)
        print("Qwen: ", response)
