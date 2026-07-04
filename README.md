<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Mental%20Health%20AI&fontSize=50&fontColor=fff&animation=fadeIn" width="100%"/>
</div>

<p align="center">
  <b>Терапевтический ИИ-помощник на русском языке</b><br>
  Собственная языковая модель, обученная с нуля на 50 000 психотерапевтических диалогах
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.6.0+CUDA-red?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/GPU-RTX_3070_8GB-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Параметры-102M-ff69b4?style=flat-square"/>
  <img src="https://img.shields.io/badge/Датасет-50_000_диалогов-success?style=flat-square"/>
  <img src="https://img.shields.io/badge/Архитектура-RoPE+SwiGLU+Flash-yellow?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square"/>
</p>

---

## О проекте

**Mental Health AI** — это языковая модель (102M параметров), обученная **с нуля** на русскоязычных психотерапевтических диалогах. Модель понимает проблемы ментального здоровья и даёт поддерживающие, эмпатичные ответы.

### Возможности

- Техники заземления при панических атаках
- Поддержка при депрессии и апатии
- Работа с тревожностью и стрессом
- Советы при проблемах с самооценкой
- Поддержка при травме и ПТСР
- Помощь при расстройствах пищевого поведения
- Информация о зависимостях
- Работа с ОКР, СДВГ, биполярным расстройством
- Кризисная поддержка

> ⚠️ **Дисклеймер**: Модель создана в образовательных целях. При серьёзных проблемах обращайтесь к профессиональным психологам и психиатрам.

---

## Архитектура

Современный декодер-only трансформер, написанный с нуля на PyTorch:

| Компонент | Описание |
|-----------|----------|
| **RoPE** | Rotary Position Embeddings — позиционная кодировка из Llama/Mistral |
| **SwiGLU** | Gated activation function (PaLM / Llama) |
| **Flash Attention** | PyTorch SDPA с аппаратным ускорением на GPU |
| **RMSNorm** | Pre-normalisation для стабильного обучения |
| **Weight Tying** | Общие веса эмбеддингов и LM головы |

### Характеристики модели

```
Параметры:  101,710,080
Embedding:  768
Layers:     14
Heads:      12
Head dim:   64
Block size: 192
Словарь:    3,382 BPE токенов
Размер:     ~400 MB (fp32)
```

---

## Быстрый старт

### 1. Установка

```bash
git clone https://github.com/Fromix1234/mental-health-app.git
cd mental-health-app
pip install torch tokenizers tqdm
```

### 2. Обучение на GPU

```bash
python train.py
```

Для CUDA (рекомендуется на RTX 3070+):
```bash
python train.py --device cuda
```

### 3. Запуск веб-интерфейса

```bash
python web_interface.py
# Открыть http://localhost:8765
```

### 4. Интерактивный чат в консоли

```bash
python generate.py
```

---

## Датасет

50 000 диалогов, охватывающих 40+ категорий психологической поддержки:

| Категория | Примеры |
|-----------|---------|
| **Тревога** | Генерализованная тревога, панические атаки, социальная тревога |
| **Депрессия** | Ангедония, апатия, потеря смысла, суицидальные мысли |
| **ПТСР и травма** | Флешбеки, кошмары, избегание |
| **ОКР** | Навязчивые мысли, компульсии, ритуалы |
| **СДВГ** | Проблемы с фокусом, прокрастинация, гипрефиксация |
| **Биполярное расстройство** | Мания, гипомания, депрессивные эпизоды |
| **РПП** | Анорексия, булимия, компульсивное переедание |
| **Зависимости** | Алкоголь, курение, цифровая зависимость |
| **Отношения** | Конфликты, расставания, созависимость |
| **Семья** | Сепарация, дисфункциональные семьи |
| **Горе и утрата** | Потеря близкого, траур, чувство вины |
| **Самооценка** | Синдром самозванца, перфекционизм |
| **Кризисы** | Четверть жизни, средний возраст, пенсия |

---

## Обучение на RTX 3070

Модель обучена с нуля на **NVIDIA RTX 3070 (8GB VRAM)** с использованием CUDA:

| Параметр | Значение |
|----------|----------|
| Batch size | 2 (gradient accumulation ×8) |
| Learning rate | 3e-4 с cosine decay |
| Warmup | 200 шагов |
| Шагов | 5000 |
| Время | ~11 минут |
| Val loss | ~4.4 (на 50K данных) |
| Использование VRAM | ~6.5 GB / 8 GB |

---

## Структура проекта

```
mental-health-app/
├── model/
│   ├── __init__.py
│   └── gpt.py              # TinyGPT: RoPE + SwiGLU + Flash Attention
├── data/
│   ├── __init__.py
│   └── dataset.py           # Генерация 50K датасета + BPE токенизатор
├── config.py                # Единый конфиг
├── train.py                 # Цикл обучения с поддержкой resume
├── generate.py              # Интерактивный чат
├── web_interface.py         # Веб-интерфейс (localhost:8765)
├── rag_search.py            # Поиск по датасету
└── requirements.txt         # Зависимости
```

---

## Технологии

- **PyTorch 2.6.0** с CUDA 12.4
- **HuggingFace Tokenizers** (BPE)
- **NVIDIA RTX 3070** для обучения
- **Flash Attention** через SDPA

---

## Планы развития

- [x] Базовая архитектура (RoPE + SwiGLU + RMSNorm)
- [x] Обучение на GPU с CUDA
- [x] 50 000+ диалогов в датасете
- [x] Веб-интерфейс
- [ ] Увеличить модель до 300M+ параметров
- [ ] GGUF-квантование для мобильных устройств
- [ ] LoRA fine-tune на Llama/Mistral
- [ ] Мобильное приложение на React Native

---

<p align="center">
  <b>Сделано с ❤️ для тех, кому нужна поддержка</b><br>
  <sub>Если проект полезен — поставьте ⭐</sub>
</p>

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=120&section=footer" width="100%"/>
</div>
