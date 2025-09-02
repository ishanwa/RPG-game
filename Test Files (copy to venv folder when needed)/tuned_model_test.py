from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load base model and tokenizer
model_name = "C:\\Users\\anjan\\Documents\\jac_lang_project\\TinyLLama_FineTune\\tinyllama"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "C:\\Users\\anjan\\Documents\\jac_lang_project\\TinyLLama_FineTune\\.venv\\tuned_model\\checkpoint-200")

# Set to evaluation mode
model.eval()


def generate_map(difficulty):
    difficulty = difficulty
    width = 20
    height = 20
    num_enemies = 8
    num_wall = 18
    prompt = f"""
    Create a game level map for a top-down 2D game with the following specifications:
    - Map size: {width}x{height} tiles
    - Level difficulty: {difficulty}/10
    - Number of enemies: {num_enemies}
    - Number of barriers: {num_wall}

    Use these symbols:
    - 'B' for barriers/walls
    - 'E' for enemies
    - 'P' for player (place at a strategic position)
    - '.' for empty walkable spaces

    The map should have:
    1. A border made of 'B' characters
    2. The player 'P' placed in a good starting position
    3. {num_enemies} enemies 'E' placed strategically
    4. {num_wall} barriers 'B' placed to create interesting paths
    5. The level should be challenging but fair for difficulty {difficulty}

    Return ONLY the map as a JSON array of strings, with no additional text.
    Example format for a 5x5 map:
    ["BBBBB", "B...B", "B.E.B", "B.P.B", "BBBBB"]
    """
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)  # ← Add padding=True

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,  # ← CRITICAL: Pass attention mask
            max_length=1024,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    map_output = response.split("### Response:")[-1].strip()
    return map_output

while True:
    difficulty = input("Enter difficulty: ")
    map_output = generate_map(difficulty)
    print(map_output)