import os
import json
import pickle
import torch
from tokenizers import Tokenizer, decoders

from config import CONFIG, ModelConfig
from model.gpt import TinyGPT


def load_model(checkpoint_path="checkpoints/best_model.pt"):
    device = CONFIG.device

    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)

    tokenizer = Tokenizer.from_file(CONFIG.data.tokenizer_file)
    tokenizer.decoder = decoders.ByteLevel()
    vocab_size = tokenizer.get_vocab_size()

    model_cfg = ModelConfig(
        vocab_size=vocab_size,
        n_embd=768,
        n_head=12,
        n_layer=14,
        block_size=192,
    )

    model = TinyGPT(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

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
            top_k=50,
            repetition_penalty=1.2,
        )

    output_ids = output_ids[0].tolist()
    response = tokenizer.decode(output_ids)

    if "Терапевт:" in response:
        response = response.split("Терапевт:", 1)[1].strip()

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
