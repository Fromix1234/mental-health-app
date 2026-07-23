import os
import sys
import json
import math
import random
import torch
from datasets import Dataset

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

try:
    from unsloth import FastLanguageModel, is_bfloat16_supported
except ImportError:
    print("Установи Unsloth: pip install unsloth")
    sys.exit(1)

from transformers import TrainingArguments
from trl import SFTTrainer

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR = "checkpoints_qwen"
GGUF_DIR = "models"
MAX_SEQ_LENGTH = 2048

def generate_dataset(num_examples=50000):
    from data.dataset import SCENARIOS, USER_PHRASES, RANDOM_MESSAGES

    pairs = list(SCENARIOS)
    data = []

    personas = [
        "Ты — опытный психолог с тёплым поддерживающим стилем.",
        "Ты — психотерапевт. Отвечай заботливо и профессионально.",
        "Ты — специалист по ментальному здоровью. Будь эмпатичен, но честен.",
    ]

    for i in range(num_examples):
        pair = random.choice(pairs)
        user_msg = pair[0].strip()
        therapist_msg = pair[1]

        if random.random() < 0.4:
            user_msg = random.choice(RANDOM_MESSAGES)
        else:
            if random.random() < 0.3:
                prefix = random.choice(USER_PHRASES) + ", "
                suffix = random.choice(["", "", " :(", " помоги", " пожалуйста"])
                user_msg = f"{prefix}{user_msg}{suffix}"

        persona = random.choice(personas)

        data.append({
            "instruction": persona,
            "input": user_msg,
            "output": therapist_msg,
        })

    return data

def format_chat(example):
    messages = [
        {"role": "system", "content": example["instruction"]},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

def main():
    data_file = "data/therapy_data.json"
    if os.path.exists(data_file):
        with open(data_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        print(f"Загружено {len(raw_data)} примеров из {data_file}")
        formatted = []
        for item in raw_data:
            text = item["text"]
            import re
            m = re.search(r"\[INST\] (.+?)\nПользователь: (.+?) \[/INST\]\nТерапевт: (.+)", text, re.DOTALL)
            if m:
                formatted.append({"instruction": m.group(1).strip(), "input": m.group(2).strip(), "output": m.group(3).strip()})
        if not formatted:
            print("Не удалось распарсить therapy_data.json, генерирую новый датасет")
            formatted = generate_dataset(50000)
    else:
        print("Генерирую датасет из 50000 диалогов...")
        formatted = generate_dataset(50000)

    print(f"Пример:\n{json.dumps(formatted[0], ensure_ascii=False, indent=2)}")

    print(f"\nЗагружаю {BASE_MODEL}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    print(f"Модель: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")

    dataset = Dataset.from_list(formatted)
    dataset = dataset.map(format_chat, remove_columns=["instruction", "input", "output"])
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            output_dir=OUTPUT_DIR,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=4,
        ),
    )

    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))

    print("\nСохраняю merged модель...")
    model.save_pretrained_merged(os.path.join(OUTPUT_DIR, "merged"), tokenizer, save_method="merged_16bit")

    print("\nЭкспортирую в GGUF Q4_K_M...")
    os.makedirs(GGUF_DIR, exist_ok=True)
    model.save_pretrained_gguf(
        GGUF_DIR,
        tokenizer,
        quantization_method="q4_k_m",
    )

    gguf_path = os.path.join(GGUF_DIR, "unsloth.Q4_K_M.gguf")
    if os.path.exists(gguf_path):
        size_gb = os.path.getsize(gguf_path) / (1024**3)
        print(f"Готово! GGUF модель: {gguf_path} ({size_gb:.2f} GB)")
    else:
        print(f"GGUF сохранён в {GGUF_DIR}/")

    print("\nГотово! Модель можно запускать: python deploy_gguf.py")

if __name__ == "__main__":
    main()
