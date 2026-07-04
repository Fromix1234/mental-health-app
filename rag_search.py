"""RAG поиск по датасету — находит подходящий ответ психолога по ключевым словам"""
import json
import os
import re
import math


_dataset = None
_pairs = []
_user_queries = []
_therapist_responses = []


def extract_pairs():
    data_file = "data/therapy_data.json"
    if not os.path.exists(data_file):
        return []
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairs = []
    for item in data:
        text = item["text"]
        user_match = re.search(r"Пользователь\s*:\s*(.+?)\s*\[/INST\]", text, re.DOTALL)
        therapist_match = re.search(r"Терапевт\s*:\s*(.+)", text, re.DOTALL)
        if user_match and therapist_match:
            pairs.append((
                user_match.group(1).strip(),
                therapist_match.group(1).strip(),
            ))
    return pairs


def tokenize(text):
    text = text.lower()
    return set(re.findall(r"[а-яё]+", text))


def load_index():
    global _pairs, _user_queries, _therapist_responses
    if _pairs:
        return
    _pairs = extract_pairs()
    _user_queries = [p[0] for p in _pairs]
    _therapist_responses = [p[1] for p in _pairs]
    print(f"RAG loaded: {len(_pairs)} dialogs")


def score(query_tokens, doc_tokens):
    if not query_tokens:
        return 0
    common = query_tokens & doc_tokens
    if not common:
        return 0
    return len(common) / math.sqrt(len(query_tokens) * len(doc_tokens))


def get_response(query):
    load_index()
    if not _pairs:
        return None, 0

    qt = tokenize(query)
    best_score = 0
    best_idx = 0

    for i, user_q in enumerate(_user_queries):
        dt = tokenize(user_q)
        s = score(qt, dt)
        if s > best_score:
            best_score = s
            best_idx = i

    if best_score < 0.1:
        return None, best_score

    return _therapist_responses[best_idx], best_score
