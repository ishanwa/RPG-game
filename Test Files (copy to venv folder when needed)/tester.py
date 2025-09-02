"""import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Get the name of the first GPU (if available)
    if num_gpus > 0:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Name: {gpu_name}")

    # Optionally, check the CUDA version linked with PyTorch
    cuda_version = torch.version.cuda
    print(f"PyTorch CUDA Version: {cuda_version}")"""

"""import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))"""

from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "C:\\Users\\anjan\\Documents\\jac_lang_project\\TinyLLama_FineTune\\tinyllama"  # or your local path

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True)
    print("✅ TinyLlama is correctly downloaded and loadable locally!")
except Exception as e:
    print("❌ Error loading TinyLlama:", e)

