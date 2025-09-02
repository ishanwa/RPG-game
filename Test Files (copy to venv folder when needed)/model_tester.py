from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Path where you saved TinyLlama (change if needed)
model_path = "C:\\Users\\anjan\\Documents\\jac_lang_project\\TinyLLama_FineTune\\tinyllama"

# ✅ Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ✅ Load model in float16 directly on GPU (RTX 4050 friendly)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map=None
).to("cuda")

# ✅ Your prompt
prompt = "How many hours does a day have ?"

# ✅ Encode input
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# ✅ Generate text
outputs = model.generate(
    **inputs,
    max_new_tokens=300,    # how long the poem can be
    do_sample=True,       # randomness for creativity
    temperature=0.8,      # creativity control
    top_p=0.9             # nucleus sampling
)

# ✅ Decode result
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Model response:\n")
print(response)
