import os
import json
import torch
from tokenizers import Tokenizer

from config import CONFIG, ModelConfig
from model.gpt import TinyGPT


def load_model(checkpoint_path="checkpoints/best_model.pt"):
    device = CONFIG.device

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    mcfg_data = ckpt["model_config"]

    model_cfg = ModelConfig(
        vocab_size=mcfg_data["vocab_size"],
        n_embd=mcfg_data["n_embd"],
        n_head=mcfg_data["n_head"],
        n_layer=mcfg_data["n_layer"],
        block_size=mcfg_data["block_size"],
    )

    model = TinyGPT(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = Tokenizer.from_file(CONFIG.data.tokenizer_file)

    return model, tokenizer, device


def generate_response(model, tokenizer, device, user_message, max_tokens=100, temperature=0.8):
    persona = "Ты — опытный психолог с теплым и поддерживающим стилем."
    prompt = f"[INST] {persona}\nПользователь: {user_message} [/INST]\nТерапевт:"

    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=40,
        )

    output_ids = output_ids[0].tolist()
    response = tokenizer.decode(output_ids)

    if "Терапевт:" in response:
        response = response.split("Терапевт:", 1)[1].strip()
    if "[/INST]" in response:
        response = response.split("[/INST]", 1)[0].strip()

    return response


def interactive():
    model, tokenizer, device = load_model()
    print("\n=== Терапевтический ИИ ===\n")
    print("Напиши что тебя беспокоит (или 'выход' для завершения)\n")

    while True:
        user_input = input("Ты: ").strip()
        if user_input.lower() in ("выход", "exit", "quit"):
            break

        response = generate_response(model, tokenizer, device, user_input)
        print(f"\nТерапевт: {response}\n")


if __name__ == "__main__":
    if not os.path.exists("checkpoints/best_model.pt"):
        print("Сначала обучи модель: python train.py")
    else:
        interactive()
