import sys, pickle, json
sys.path.insert(0, r"M:\python312_libs")
sys.path.insert(0, r"M:\mental-health-app")

from tokenizers import Tokenizer
from config import ModelConfig
from model.gpt import TinyGPT
import torch

device = "cuda"
print("Loading model...")
with open("checkpoints/best_model.pt", "rb") as f:
    ckpt = pickle.load(f)

tokenizer = Tokenizer.from_file("data/tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab: {vocab_size}")

cfg = ModelConfig(vocab_size=vocab_size, n_embd=768, n_head=12, n_layer=14, block_size=192)
model = TinyGPT(cfg)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device)
model.eval()

def chat(user_msg):
    prompt = f"[INST] Ты опытный психолог. Будь теплым и поддерживающим.\nПользователь: {user_msg} [/INST]\nТерапевт:"
    encoded = tokenizer.encode(prompt)
    inp = torch.tensor([encoded.ids], dtype=torch.long).to(device)
    with torch.no_grad():
        out = model.generate(inp, max_new_tokens=100, temperature=0.8, top_k=50, repetition_penalty=1.2)
    resp = tokenizer.decode(out[0].tolist())
    if "Терапевт:" in resp:
        resp = resp.split("Терапевт:", 1)[1].strip()
    return resp

tests = [
    "У меня паническая атака, что делать?",
    "Я чувствую себя одиноко",
    "Меня никто не понимает",
    "Я боюсь будущего",
]
results = []
for q in tests:
    a = chat(q)
    results.append({"user": q, "therapist": a})
    print(f"\nТы: {q}")
    print(f"Терапевт: {a}")

with open(r"M:\mental-health-app\test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n\nРезультаты сохранены в test_results.json")

