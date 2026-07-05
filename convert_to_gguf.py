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
GGUF_TYPE_ARRAY = 7


def write_kv_string(f, key, value):
    key_bytes = key.encode("utf-8")
    f.write(struct.pack("<I", len(key_bytes)))
    f.write(key_bytes)
    f.write(struct.pack("<I", GGUF_TYPE_STRING))
    val_bytes = value.encode("utf-8")
    f.write(struct.pack("<Q", len(val_bytes)))
    f.write(val_bytes)


def write_kv_uint32(f, key, value):
    key_bytes = key.encode("utf-8")
    f.write(struct.pack("<I", len(key_bytes)))
    f.write(key_bytes)
    f.write(struct.pack("<I", GGUF_TYPE_UINT32))
    f.write(struct.pack("<I", value))


def write_kv_float32(f, key, value):
    key_bytes = key.encode("utf-8")
    f.write(struct.pack("<I", len(key_bytes)))
    f.write(key_bytes)
    f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
    f.write(struct.pack("<f", value))


def write_kv_string_array(f, key, items):
    key_bytes = key.encode("utf-8")
    f.write(struct.pack("<I", len(key_bytes)))
    f.write(key_bytes)
    f.write(struct.pack("<I", GGUF_TYPE_ARRAY))
    f.write(struct.pack("<I", GGUF_TYPE_STRING))
    f.write(struct.pack("<Q", len(items)))
    for item in items:
        val_bytes = item.encode("utf-8")
        f.write(struct.pack("<Q", len(val_bytes)))
        f.write(val_bytes)


def write_kv_float32_array(f, key, items):
    key_bytes = key.encode("utf-8")
    f.write(struct.pack("<I", len(key_bytes)))
    f.write(key_bytes)
    f.write(struct.pack("<I", GGUF_TYPE_ARRAY))
    f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
    f.write(struct.pack("<Q", len(items)))
    for item in items:
        f.write(struct.pack("<f", item))


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


def _load_tokenizer_vocab(tokenizer_path="data/tokenizer.json"):
    try:
        from tokenizers import Tokenizer
        t = Tokenizer.from_file(tokenizer_path)
        vocab = t.get_vocab()
        sorted_tokens = [""] * len(vocab)
        for token, tid in vocab.items():
            if tid < len(sorted_tokens):
                sorted_tokens[tid] = token
        return sorted_tokens
    except Exception as e:
        print(f"  Предупреждение: не удалось загрузить токенизатор ({e})")
        return None


def convert_to_gguf(ckpt_path, output_path, tokenizer_path="data/tokenizer.json"):
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
    head_dim = n_embd // n_head
    rope_dim = head_dim * 2

    print(f"  n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}, n_ff={n_ff}, rope_dim={rope_dim}")

    tokens = _load_tokenizer_vocab(tokenizer_path)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))

        tensor_count = (
            1 + 1 + 1 + n_layer * 9
        )
        f.write(struct.pack("<Q", tensor_count))

        kv = 7
        if tokens:
            kv += 7

        f.write(struct.pack("<Q", kv))

        write_kv_string(f, "general.architecture", "llama")
        write_kv_string(f, "general.name", "Mental Health AI")
        write_kv_uint32(f, "llama.context_length", block_size)
        write_kv_uint32(f, "llama.embedding_length", n_embd)
        write_kv_uint32(f, "llama.block_count", n_layer)
        write_kv_uint32(f, "llama.head_count", n_head)
        write_kv_uint32(f, "llama.head_count_kv", n_head)
        write_kv_uint32(f, "llama.feed_forward_length", n_ff)
        write_kv_float32(f, "llama.attention.layer_norm_rms_epsilon", 1e-6)
        write_kv_uint32(f, "llama.rope.dimension_count", rope_dim)
        write_kv_float32(f, "llama.rope.freq_base", 10000.0)
        write_kv_uint32(f, "general.file_type", 0)

        if tokens:
            write_kv_string(f, "tokenizer.ggml.model", "llama")
            write_kv_string_array(f, "tokenizer.ggml.tokens", tokens)
            scores = [0.0] * len(tokens)
            write_kv_float32_array(f, "tokenizer.ggml.scores", scores)
            write_kv_uint32(f, "tokenizer.ggml.bos_token_id", 2)
            write_kv_uint32(f, "tokenizer.ggml.eos_token_id", 3)
            write_kv_uint32(f, "tokenizer.ggml.padding_token_id", 0)
            write_kv_uint32(f, "tokenizer.ggml.add_bos_token", 1)

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
    tkn_path = "data/tokenizer.json"
    if not os.path.exists(tkn_path):
        tkn_path = None
    convert_to_gguf(args.checkpoint, args.output, tokenizer_path=tkn_path)


if __name__ == "__main__":
    main()
