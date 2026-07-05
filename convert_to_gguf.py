#!/usr/bin/env python3
"""Конвертация TinyGPT в GGUF формат для llama.cpp"""
import os
import sys
import json
import struct
import numpy as np

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

GGUF_KEY_TOKENIZER_MODEL = 401
GGUF_KEY_CONTEXT_LENGTH = 404
GGUF_KEY_EMBEDDING_LENGTH = 405
GGUF_KEY_BLOCK_COUNT = 406
GGUF_KEY_HEAD_COUNT = 407
GGUF_KEY_LAYER_NORM_RMS_EPS = 408
GGUF_KEY_FEED_FORWARD_LENGTH = 409
GGUF_KEY_ATTENTION_LAYERNORM_RMS_EPS = 415
GGUF_KEY_ROPE_FREQ_BASE = 416
GGUF_KEY_ROPE_SCALE_LINEAR = 417

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 8
GGUF_TYPE_BOOL = 9
GGUF_TYPE_STRING = 6
GGUF_TYPE_ARRAY = 7
GGUF_TYPE_FLOAT64 = 10

TENSOR_NAMES = {
    "token_embedding.weight": "token_embd.weight",
    "ln_f.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}


def gguf_quantize_f32(data):
    return data.astype(np.float32)


def write_gguf_tensor(f, name, tensor):
    name_bytes = name.encode("utf-8") + b"\x00"
    f.write(struct.pack("<I", len(name_bytes)))
    f.write(name_bytes)
    shape = tensor.shape
    f.write(struct.pack("<I", len(shape)))
    for dim in reversed(shape):
        f.write(struct.pack("<Q", dim))
    f.write(struct.pack("<I", 0))
    f.write(tensor.tobytes())


def convert_to_gguf(ckpt_path, output_path, vocab_size=10000):
    print(f"Загружаю чекпоинт: {ckpt_path}")
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    sd = ckpt["model_state_dict"]
    mcfg = ckpt.get("model_config", {})
    n_embd = mcfg.get("n_embd", 384)
    n_head = mcfg.get("n_head", 8)
    n_layer = mcfg.get("n_layer", 8)
    block_size = mcfg.get("block_size", 256)

    head_dim = n_embd // n_head
    n_ff = int(8 * n_embd / 3)

    print(f"  n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}, n_ff={n_ff}")

    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))

        tensor_count = (
            1 +  # token_embd
            1 +  # output_norm
            1 +  # output
            n_layer * 7  # per layer: attn_norm, q, k, v, attn_output, ffn_norm, ffn_gate, ffn_up, ffn_down
        )
        f.write(struct.pack("<Q", tensor_count))

        kv_count = 10
        f.write(struct.pack("<Q", kv_count))

        def write_kv(key, ktype, value):
            key_bytes = key.encode("utf-8")
            f.write(struct.pack("<I", len(key_bytes)))
            f.write(key_bytes)
            f.write(struct.pack("<I", ktype))
            if ktype == GGUF_TYPE_UINT32:
                f.write(struct.pack("<I", value))
            elif ktype == GGUF_TYPE_FLOAT32:
                f.write(struct.pack("<f", value))
            elif ktype == GGUF_TYPE_BOOL:
                f.write(struct.pack("<?", value))
            elif ktype == GGUF_TYPE_FLOAT64:
                f.write(struct.pack("<d", value))
            elif ktype == GGUF_TYPE_STRING:
                val_bytes = value.encode("utf-8")
                f.write(struct.pack("<Q", len(val_bytes)))
                f.write(val_bytes)

        write_kv("general.architecture", GGUF_TYPE_STRING, "llama")
        write_kv("general.name", GGUF_TYPE_STRING, "Mental Health AI")
        write_kv(GGUF_KEY_TOKENIZER_MODEL, GGUF_TYPE_STRING, "no_tokenizer")
        write_kv(GGUF_KEY_CONTEXT_LENGTH, GGUF_TYPE_UINT32, block_size)
        write_kv(GGUF_KEY_EMBEDDING_LENGTH, GGUF_TYPE_UINT32, n_embd)
        write_kv(GGUF_KEY_BLOCK_COUNT, GGUF_TYPE_UINT32, n_layer)
        write_kv(GGUF_KEY_HEAD_COUNT, GGUF_TYPE_UINT32, n_head)
        write_kv(GGUF_KEY_FEED_FORWARD_LENGTH, GGUF_TYPE_UINT32, n_ff)
        write_kv(GGUF_KEY_LAYER_NORM_RMS_EPS, GGUF_TYPE_FLOAT32, 1e-6)
        write_kv("general.file_type", GGUF_TYPE_UINT32, 0)

        rope_freq = mcfg.get("rope_freq", 10000.0)
        write_kv(GGUF_KEY_ROPE_FREQ_BASE, GGUF_TYPE_FLOAT32, rope_freq)

        print("  Пишу тензоры...")

        te = sd["token_embedding.weight"].numpy().astype(np.float32)
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
