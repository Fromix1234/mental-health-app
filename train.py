import os
import math
import time
import pickle
import tempfile
tempfile.tempdir = r"M:\temp"
import torch
from torch.utils.data import Dataset, DataLoader

from config import CONFIG, ModelConfig


def save_checkpoint(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
from model.gpt import TinyGPT
from data.dataset import load_or_create_dataset


class TherapyDataset(Dataset):
    def __init__(self, data, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.encoded = []

        for item in data:
            ids = tokenizer.encode(item["text"]).ids
            if len(ids) < 2:
                continue
            ids = ids[:block_size + 1]
            self.encoded.append(ids)

        print(f"Sequences: {len(self.encoded)}")

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        tokens = self.encoded[idx]
        x = torch.full((self.block_size,), self.pad_id, dtype=torch.long)
        y = torch.full((self.block_size,), -1, dtype=torch.long)
        seq_len = min(len(tokens) - 1, self.block_size)
        x[:seq_len] = torch.tensor(tokens[:-1][:seq_len], dtype=torch.long)
        y[:seq_len] = torch.tensor(tokens[1:][:seq_len], dtype=torch.long)
        return x, y


@torch.no_grad()
def estimate_loss(model, dataloader, device):
    model.eval()
    losses = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def get_lr(it, cfg):
    warmup = cfg.training.warmup_steps
    max_steps = cfg.training.max_steps
    lr = cfg.training.learning_rate
    if it < warmup:
        return lr * it / warmup
    decay = (it - warmup) / (max_steps - warmup)
    return lr * 0.5 * (1.0 + math.cos(math.pi * decay))


def main():
    import sys
    resume = "--resume" in sys.argv

    cfg = CONFIG
    os.makedirs(cfg.output_dir, exist_ok=True)

    data, tokenizer = load_or_create_dataset()

    split_idx = int(len(data) * cfg.data.train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    train_dataset = TherapyDataset(train_data, tokenizer, cfg.model.block_size)
    val_dataset = TherapyDataset(val_data, tokenizer, cfg.model.block_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )

    model_cfg = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        n_embd=cfg.model.n_embd,
        n_head=cfg.model.n_head,
        n_layer=cfg.model.n_layer,
        block_size=cfg.model.block_size,
        dropout=cfg.model.dropout,
        use_checkpoint=cfg.model.use_checkpoint,
    )

    model = TinyGPT(model_cfg)
    model.to(cfg.device)

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Device: {cfg.device}")
    print(f"Batch: {cfg.training.batch_size} x {cfg.training.grad_accum_steps} = {cfg.training.batch_size * cfg.training.grad_accum_steps}")
    print(f"Max steps: {cfg.training.max_steps}")

    optimizer = model.configure_optimizers(
        weight_decay=cfg.training.weight_decay,
        learning_rate=cfg.training.learning_rate,
        betas=(cfg.training.beta1, cfg.training.beta2),
        device_type="cuda" if cfg.device == "cuda" else "cpu",
    )

    model.train()
    best_val_loss = float("inf")
    step = 0
    tokens_processed = 0
    total_start_time = time.time()

    if resume:
        ckpt_path = os.path.join(cfg.output_dir, "best_model.pt")
        if os.path.exists(ckpt_path):
            ckpt = pickle.load(open(ckpt_path, "rb"))
            model.load_state_dict(ckpt["model_state_dict"])
            step = ckpt["step"] + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))
            total_start_time = time.time()
            print(f"Resumed from step {step} (best val_loss: {best_val_loss:.4f})")
        else:
            print(f"No checkpoint found at {ckpt_path}, starting from scratch")

    for epoch in range(100):
        if step >= cfg.training.max_steps:
            break
        for x, y in train_loader:
            if step >= cfg.training.max_steps:
                break

            lr = get_lr(step, cfg)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x, y = x.to(cfg.device), y.to(cfg.device)
            _, loss = model(x, y)
            loss = loss / cfg.training.grad_accum_steps
            loss.backward()

            tokens_processed += x.numel()

            if (step + 1) % cfg.training.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if step % cfg.training.log_interval == 0:
                tokens_per_sec = tokens_processed / (time.time() - total_start_time)
                msg = (f"step {step:5d} | loss {loss.item() * cfg.training.grad_accum_steps:.4f} | "
                       f"lr {lr:.2e} | tok/s {tokens_per_sec:.0f}")
                with open(cfg.log_file, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                print(msg)

            if step % cfg.training.eval_interval == 0:
                torch.cuda.empty_cache()
                val_loss = estimate_loss(model, val_loader, cfg.device)
                msg = f"step {step:5d} | val_loss {val_loss:.4f}"
                with open(cfg.log_file, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
                print(msg)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    final_path = os.path.join(cfg.output_dir, "best_model.pt")
                    save_checkpoint(
                        {
                            "step": step,
                            "model_state_dict": model.state_dict(),
                            "val_loss": val_loss,
                        },
                        final_path,
                    )
                    print(f"  -> Saved best model (val_loss: {val_loss:.4f})")

            if step % cfg.training.save_interval == 0 and step > 0:
                final_path = os.path.join(cfg.output_dir, f"checkpoint_{step}.pt")
                save_checkpoint(
                    {
                        "step": step,
                        "model_state_dict": model.state_dict(),
                    },
                    final_path,
                )

            step += 1

    final_path = os.path.join(cfg.output_dir, "final_model.pt")
    save_checkpoint(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
        },
        final_path,
    )

    total_time = time.time() - total_start_time
    print(f"\nTraining done! Time: {total_time:.1f}s")
    print(f"Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
