import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import json
from datasets import Dataset, load_from_disk
model_name = "C:\\Users\\anjan\\Documents\\jac_lang_project\\TinyLLama_FineTune\\tinyllama"
tokenized_dataset = "C:\\Users\\anjan\\Documents\\jac_lang_project\\TinyLLama_FineTune\\.venv\\tokenized_dataset"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, # Or load in 8bit -> Defines the bit config for matrix coefficients
    bnb_4bit_use_double_quant=True, # Applies extra round of quantization to save memory
    bnb_4bit_quant_type='nf4', # 4bit quantization algorithm (nf4 or pf4, nf4 better for LLMs)
    bnb_4bit_compute_dtype='float16'
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map='auto')

#Loading the tokenizer -> TinyLLama itself to tokenize the wording
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS token

#---------------------------------------------------------------------------------------------------------------------
# Freezing the Original weights -> Setting W constant
#---------------------------------------------------------------------------------------------------------------------

for param in model.parameters():
    param.requires_grad = False # Freezing the model -> grad(W) = 0
    if param.ndim == 1:
        # Casts the small parameters (layer norm) to fp32 for stability
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable() #Reduces number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

#---------------------------------------------------------------------------------------------------------------------
# Setting up LoRA adapters
#---------------------------------------------------------------------------------------------------------------------
def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        trainable_params += param.numel()
    print(f"Trainable parameters: {trainable_params}"
          f"|| Total parameters: {all_params} || Trainable %: {trainable_params / all_params * 100:.2f}%")

print_trainable_parameters(model)


config = LoraConfig(
    r=16, #Attention Heads -> No. of rows in A and columns in B
    lora_alpha=32, # Scaling Factor
    lora_dropout=0.05, # 5% of input embeddings will be zeroed randomly
    bias='none', # y = Wx + b -> b=0
    task_type="CAUSAL_LM" # Model learns to predict the next token
    # ex: John wick killed 3 people in a bar with a f**** -> "Pencil"
)

#---------------------------------------------------------------------------------------------------------------------
# Loading the dataset and model
#---------------------------------------------------------------------------------------------------------------------

model = get_peft_model(model, config)
print("Model is ready for finetuning!")
tokenized_data = load_from_disk(tokenized_dataset)

#---------------------------------------------------------------------------------------------------------------------
# Setting up trainer
#---------------------------------------------------------------------------------------------------------------------


trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=16, # Processes 16 samples per GPU at once
        gradient_accumulation_steps=10, # Updates weights after each n steps, high -> smooth, low -> noisy
        warmup_steps=100, # gradually increases learning rate until first 100 steps -> high learning stability
        max_steps=200, # High -> Better learning, tendency to overfit.
        learning_rate=2e-4,
        fp16=True, #Cuts memory usage in half
        output_dir='C:\\Users\\harsh\\PycharmProjects\\TinyLLama_FineTune\\.venv\\tuned_model',
        logging_steps=1, # Prints progress each and every step
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
)
print("Trainer Ready.")

#---------------------------------------------------------------------------------------------------------------------
# Model Training
#---------------------------------------------------------------------------------------------------------------------

model.config.use_cache=False
trainer.train()


print("Fine tuning completed and model saved!")