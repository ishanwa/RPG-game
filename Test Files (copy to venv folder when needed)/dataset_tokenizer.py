import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import load_from_disk

# Tokenizing the dataset



from datasets import load_from_disk

tokenizer = AutoTokenizer.from_pretrained("C:\\Users\\anjan\\Documents\\jac_lang_project\\TinyLLama_FineTune\\tinyllama")
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # Tokenize the entire text
    tokenized = tokenizer(
        examples["sample"],
        truncation=True,
        max_length=1024,  # Adjust based on your longest map
        padding=True,
        return_tensors=None
    )

    # For causal LM, labels are the same as input_ids
    # The model learns to predict the next token in the sequence
    tokenized["labels"] = tokenized["input_ids"].copy()
    # Convert all lists to tensors
    for key in tokenized:
        tokenized[key] =  torch.tensor(tokenized[key]) # Keep as lists, HuggingFace handles conversion

    return tokenized


# Load the saved dataset
loaded_dataset = load_from_disk("C:\\Users\\anjan\\Documents\\jac_lang_project\\TinyLLama_FineTune\\.venv\\formatted_dataset_new")
print("Loaded successfully")
# Now you can use it for tokenization and training
tokenized_dataset = loaded_dataset.map(
    tokenize_function,
    remove_columns=loaded_dataset.column_names,
    batched=True
)
tokenized_dataset.save_to_disk("C:\\Users\\anjan\\Documents\\jac_lang_project\\TinyLLama_FineTune\\.venv\\tokenized_dataset")
print("Tokenized dataset saved to 'tokenized_dataset' directory")


tokenized_dataset = load_from_disk("C:\\Users\\anjan\\Documents\\jac_lang_project\TinyLLama_FineTune\\.venv\\tokenized_dataset")
print("Tokenized dataset loaded from 'tokenized_dataset' directory")


# Verify a sample
sample = tokenized_dataset[0]
print("Sample input IDs:", sample['input_ids'][:20])
print("Sample decoded:")
print(tokenizer.decode(sample['input_ids'], skip_special_tokens=True))