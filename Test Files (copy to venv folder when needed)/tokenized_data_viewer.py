import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
tokenized_dataset = load_from_disk("C:\\Users\\anjan\\Documents\\jac_lang_project\\TinyLLama_FineTune\\.venv\\tokenized_dataset")
model_name = "C:\\Users\\anjan\\Documents\\jac_lang_project\\TinyLLama_FineTune\\tinyllama"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# View a few samples
for i in range(5):  # First 3 examples
    sample = tokenized_dataset[i]
    print(f"\n=== Sample {i} ===")
    print("Input IDs:", sample['input_ids'][:20], "...")  # First 20 tokens
    print("Attention Mask:", sample['attention_mask'][:20], "...")
    print("Labels:", sample['labels'][:20], "...")

    # Decode back to text
    decoded_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
    print("Decoded text:")
    print(decoded_text)
    print("-" * 50)