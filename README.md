<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Mental%20Health%20AI&fontSize=50&fontColor=fff&animation=fadeIn" width="100%"/>
</div>

<p align="center">
  <b>Терапевтический ИИ-помощник на русском языке</b><br>
  Собственная языковая модель, обученная с нуля для психологической поддержки
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/RTX_3070-✅_8GB_VRAM-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Параметры-5.3M-ff69b4?style=flat-square"/>
  <img src="https://img.shields.io/badge/Датасет-5000_диалогов-success?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>
</p>

---

## О проекте

**Mental Health AI** — это крошечная, но полноценная языковая модель, обученная **с нуля** на русскоязычных психотерапевтических диалогах. Модель понимает проблемы ментального здоровья и даёт поддерживающие, эмпатичные ответы в стиле профессионального психолога.

### Возможности

- Техники заземления при панических атаках
- Поддержка при депрессии и апатии
- Работа с тревожностью и стрессом
- Советы при проблемах с самооценкой
- Поддержка при травме и ПТСР
- Помощь при расстройствах пищевого поведения
- Информация о зависимостях

> ⚠️ **Дисклеймер**: Модель создана в образовательных целях. При серьёзных проблемах обращайтесь к профессиональным психологам и психиатрам.

---

## Архитектура

Современный декодер-only трансформер с нуля:

| Компонент | Детали |
|-----------|--------|
| **RoPE** | Rotary Position Embeddings — лучшая позиционная кодировка |
| **SwiGLU** | Gated activation function из PaLM / Llama |
| **Flash Attention** | PyTorch SDPA с аппаратным ускорением |
| **RMSNorm** | Pre-norm стабилизация из Llama/Mistral |
| **Weight Tying** | Общие веса эмбеддингов и LM головы |

```
Параметры:  5,332,224
Embedding:  256
Layers:     6
Heads:      8
Block size: 192
```

---

## Быстрый старт

```bash
# Клонировать
git clone https://github.com/Fromix1234/mental-health-app.git
cd mental-health-app

# Установить зависимости
pip install torch tokenizers tqdm

# Обучить модель (настроено под 8GB VRAM)
python train.py

# Начать диалог
python generate.py
```

### Разговор с моделью

```
=== Терапевтический ИИ ===

Ты: Мне очень тревожно, помоги успокоиться
Терапевт: Попробуй технику заземления: назови 5 вещей, которые
видишь, 4 — чувствуешь, 3 — слышишь, 2 — нюхаешь, 1 — пробуешь на вкус.

Ты: Я чувствую себя одиноко
Терапевт: Одиночество — это разрыв между реальными и желаемыми
связями. Начни с малого: улыбнись бариста, скажи комплимент коллеге.
```

---

## Датасет

5000 синтетических диалогов, охватывающих:

- Тревожные расстройства
- Депрессию и апатию
- Панические атаки
- Одиночество
- Проблемы с самооценкой
- Выгорание и стресс
- Горе и утрату
- ПТСР и травму
- Отношения
- Расстройства пищевого поведения
- Зависимости
- Гнев и эмоциональную регуляцию

---

## Обучение на RTX 3070

Модель обучена с нуля на **NVIDIA RTX 3070 (8GB VRAM)**:

| Параметр | Значение |
|----------|----------|
| Batch size | 4 (gradient accumulation ×4) |
| Learning rate | 3e-4 с cosine decay |
| Warmup | 200 шагов |
| Шагов | 5000 |
| Время | ~16 минут |
| Val loss | 0.0410 |

Современные техники (Gradient clipping, AdamW) для стабильного обучения на ограниченном VRAM.

---

## Структура проекта

```
mental-health-app/
├── model/
│   ├── __init__.py
│   └── gpt.py          # TinyGPT: RoPE + SwiGLU + Flash Attention
├── data/
│   ├── __init__.py
│   └── dataset.py       # Генерация датасета и BPE-токенизатор
├── config.py            # Единый конфиг
├── train.py             # Цикл обучения
├── generate.py          # Интерактивный чат
└── requirements.txt     # Зависимости
```

---

## Планы развития

- [ ] Увеличить модель (n_embd=512, ~30M params)
- [ ] Добавить реальные диалоги с психотерапии
- [ ] GGUF-квантование для запуска на телефоне
- [ ] React Native мобильное приложение
- [ ] LoRA fine-tune на Llama 3 / Mistral

---

<p align="center">
  <b>Сделано с ❤️ для тех, кому нужна поддержка</b><br>
  <sub>Если проект полезен — поставь ⭐</sub>
</p>

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer" width="100%"/>
</div>
