#!/usr/bin/env python3
"""Конвертация TinyGPT в GGUF формат для llama.cpp"""
import os
import sys
import struct

import numpy as np

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 8
GGUF_TYPE_STRING = 6


def write_gguf_tensor(f, name, tensor):
    data = tensor.astype(np.float32)
    name_bytes = name.encode("utf-8") + b"\x00"
    f.write(struct.pack("<I", len(name_bytes)))
    f.write(name_bytes)
    shape = data.shape
    f.write(struct.pack("<I", len(shape)))
    for dim in reversed(shape):
        f.write(struct.pack("<Q", dim))
    f.write(struct.pack("<I", 0))
    f.write(data.tobytes())


def convert_to_gguf(ckpt_path, output_path):
    print(f"Загружаю чекпоинт: {ckpt_path}")
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    sd = ckpt["model_state_dict"]
    mcfg = ckpt.get("model_config", {})
    n_embd = mcfg.get("n_embd", 384)
    n_head = mcfg.get("n_head", 8)
    n_layer = mcfg.get("n_layer", 8)
    block_size = mcfg.get("block_size", 256)

    n_ff = int(8 * n_embd / 3)

    print(f"  n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}, n_ff={n_ff}")

    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))

        tensor_count = (
            1 + 1 + 1 + n_layer * 9
        )
        f.write(struct.pack("<Q", tensor_count))

        kv_data = [
            ("general.architecture", GGUF_TYPE_STRING, "llama"),
            ("general.name", GGUF_TYPE_STRING, "Mental Health AI"),
            ("llama.context_length", GGUF_TYPE_UINT32, block_size),
            ("llama.embedding_length", GGUF_TYPE_UINT32, n_embd),
            ("llama.block_count", GGUF_TYPE_UINT32, n_layer),
            ("llama.head_count", GGUF_TYPE_UINT32, n_head),
            ("llama.feed_forward_length", GGUF_TYPE_UINT32, n_ff),
            ("llama.attention.layer_norm_rms_epsilon", GGUF_TYPE_FLOAT32, 1e-6),
            ("general.file_type", GGUF_TYPE_UINT32, 0),
        ]

        f.write(struct.pack("<Q", len(kv_data)))
        for key, ktype, value in kv_data:
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("<I", len(key_bytes)))
            f.write(key_bytes)
            f.write(struct.pack("<I", ktype))
            if ktype == GGUF_TYPE_UINT32:
                f.write(struct.pack("<I", value))
            elif ktype == GGUF_TYPE_FLOAT32:
                f.write(struct.pack("<f", value))
            elif ktype == GGUF_TYPE_STRING:
                val_bytes = value.encode("utf-8")
                f.write(struct.pack("<Q", len(val_bytes)))
                f.write(val_bytes)

        print("  Пишу тензоры...")

        te = sd["token_embedding.weight"].numpy()
        write_gguf_tensor(f, "token_embd.weight", te)

        for i in range(n_layer):
            prefix_attn = f"blocks.{i}.attn"
            prefix_mlp = f"blocks.{i}.mlp"

            ln1 = sd[f"blocks.{i}.ln_1.weight"].numpy().astype(np.float32)
            write_gguf_tensor(f, f"blk.{i}.attn_norm.weight", ln1)

            c_attn = sd[f"{prefix_attn}.c_attn.weight"].numpy().astype(np.float32)
            q, k, v = np.split(c_attn, 3, axis=0)
            write_gguf_tensor(f, f"blk.{i}.attn_q.weight", q)
            write_gguf_tensor(f, f"blk.{i}.attn_k.weight", k)
            write_gguf_tensor(f, f"blk.{i}.attn_v.weight", v)

            c_proj = sd[f"{prefix_attn}.c_proj.weight"].numpy().astype(np.float32)
            write_gguf_tensor(f, f"blk.{i}.attn_output.weight", c_proj)

            ln2 = sd[f"blocks.{i}.ln_2.weight"].numpy().astype(np.float32)
            write_gguf_tensor(f, f"blk.{i}.ffn_norm.weight", ln2)

            gate = sd[f"{prefix_mlp}.gate_proj.weight"].numpy().astype(np.float32)
            write_gguf_tensor(f, f"blk.{i}.ffn_gate.weight", gate)

            up = sd[f"{prefix_mlp}.up_proj.weight"].numpy().astype(np.float32)
            write_gguf_tensor(f, f"blk.{i}.ffn_up.weight", up)

            down = sd[f"{prefix_mlp}.down_proj.weight"].numpy().astype(np.float32)
            write_gguf_tensor(f, f"blk.{i}.ffn_down.weight", down)

            if (i + 1) % 2 == 0 or i == n_layer - 1:
                print(f"    слой {i+1}/{n_layer}")

        ln_f = sd["ln_f.weight"].numpy().astype(np.float32)
        write_gguf_tensor(f, "output_norm.weight", ln_f)

        output = sd["lm_head.weight"].numpy().astype(np.float32)
        write_gguf_tensor(f, "output.weight", output)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nГотово! {output_path} ({size_mb:.1f} MB)")
    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Конвертация TinyGPT в GGUF")
    parser.add_argument("checkpoint", nargs="?", default="checkpoints/best_model.pt",
                        help="Путь к .pt чекпоинту")
    parser.add_argument("--output", "-o", default="models/mental_health_ai.gguf",
                        help="Выходной GGUF файл")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Чекпоинт не найден: {args.checkpoint}")
        print("Сначала обучи модель: python train.py")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    convert_to_gguf(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
