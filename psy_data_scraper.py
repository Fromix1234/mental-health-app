import json
import re
import time
import hashlib
import requests
from urllib.parse import urljoin, urlparse
from pathlib import Path


def anonymize(text):
    text = re.sub(r"\b\d{11}\b", "[НОМЕР]", text)
    text = re.sub(
        r"[А-ЯЁ][а-яё]+ [А-ЯЁ]\.[А-ЯЁ]\.", "[ИМЯ]", text
    )
    text = re.sub(
        r"@\w+", "[@НИК]", text
    )
    text = re.sub(
        r"(https?://|www\.)\S+", "[ССЫЛКА]", text
    )
    text = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", text
    )
    return text


def make_id(url):
    return hashlib.md5(url.encode()).hexdigest()[:12]


def safe_get(url, headers, retries=2):
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            time.sleep(0.5)
            return resp.text
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
    return None


def scrape_b17_forum():
    headers = {
        "User-Agent": "MentalHealthBot/1.0 (educational research; contact: fromix@example.com)",
        "Accept": "text/html,application/xhtml+xml",
    }
    seen_ids = set()
    results = []

    forum_urls = [
        "https://www.b17.ru/forum/",
    ]

    for forum_url in forum_urls:
        html = safe_get(forum_url, headers)
        if not html:
            continue

        topic_urls = re.findall(
            r'href="(/forum/[a-zA-Z0-9_-]+/)"', html
        )
        topic_urls = list(dict.fromkeys(topic_urls))[:50]

        for topic_rel in topic_urls:
            topic_url = urljoin("https://www.b17.ru", topic_rel)
            topic_html = safe_get(topic_url, headers)
            if not topic_html:
                continue

            title_match = re.search(
                r'<title>(.*?)</title>', topic_html, re.DOTALL
            )
            title = title_match.group(1).strip() if title_match else ""

            posts = re.findall(
                r'<div class="post-message[^"]*">(.*?)</div>',
                topic_html, re.DOTALL,
            )
            for post_html in posts:
                text = re.sub(r"<[^>]+>", "", post_html)
                text = re.sub(r"\s+", " ", text).strip()
                text = anonymize(text)
                if len(text) < 100:
                    continue
                text_id = make_id(text)
                if text_id in seen_ids:
                    continue
                seen_ids.add(text_id)
                results.append({
                    "id": text_id,
                    "source": "b17.ru",
                    "text": f"Пользователь: {text}\nТерапевт: Я вас слышу. Расскажите подробнее, что вас беспокоит?",
                })

    print(f"b17.ru: {len(results)} постов")
    return results


def scrape_psychology_forums():
    headers = {
        "User-Agent": "MentalHealthBot/1.0 (educational research)",
        "Accept": "text/html,application/xhtml+xml",
    }
    results = []

    sources = [
        {
            "name": "psychologies.ru",
            "forum_urls": [
                "https://www.psychologies.ru/forum/",
            ],
            "post_pattern": r'<p class="forum-post[^"]*">(.*?)</p>',
        },
    ]

    for source in sources:
        for forum_url in source["forum_urls"]:
            html = safe_get(forum_url, headers)
            if not html:
                continue

            posts = re.findall(
                source["post_pattern"], html, re.DOTALL
            )
            for post_html in posts[:100]:
                text = re.sub(r"<[^>]+>", "", post_html)
                text = re.sub(r"\s+", " ", text).strip()
                text = anonymize(text)
                if len(text) < 80:
                    continue
                text_id = make_id(text)
                results.append({
                    "id": text_id,
                    "source": source["name"],
                    "text": f"Пользователь: {text}\nТерапевт: Понимаю вас. Это действительно сложная ситуация. Что вы чувствуете?",
                })

    print(f"psychologies.ru: {len(results)} постов")
    return results


def generate_crisis_dialogues():
    templates = [
        ("Мне кажется, я больше не могу так жить. Всё потеряло смысл.", "Мне очень жаль, что вы чувствуете себя так. Пожалуйста, помните, что вы не одиноки. Если вам нужна срочная помощь, позвоните по номеру 8-800-2000-122 — это круглосуточная линия доверия. Не игнорируйте эти чувства, обратитесь за профессиональной помощью."),
        ("Я боюсь, что не справлюсь с экзаменами, у меня паника", "Тревога перед экзаменами — это нормально. Попробуйте технику 4-7-8: вдох на 4 счёта, задержка на 7, выдох на 8. Сделайте 5 циклов. Вы подготовились и у вас всё получится."),
        ("Меня бросил парень, и я чувствую себя пустым местом", "Расставание — это тяжело. Ваша ценность не зависит от того, рядом с вами кто-то или нет. Позвольте себе погрустить, но помните: это временно."),
        ("Я constantly сравниваю себя с другими и чувствую себя никчёмным", "Сравнение — вор радости. Сосредоточьтесь на своём прогрессе, а не на чужом. Напишите 3 вещи, которые у вас получились сегодня."),
        ("У меня паническая атака прямо сейчас, что делать?", "Сядьте удобно. Назовите 5 вещей, которые видите, 4 — которые можете потрогать, 3 — слышите, 2 — чувствуете запах, 1 — можете попробовать. Дышите медленно. Это пройдёт."),
    ]
    return [
        {
            "id": make_id(q),
            "source": "crisis_template",
            "text": f"Пользователь: {q}\nТерапевт: {a}",
        }
        for q, a in templates
    ]


def save_and_merge(new_dialogs, output_path="data/therapy_data.json"):
    output_path = Path(output_path)

    existing = []
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    existing_ids = {d["id"] for d in existing if "id" in d}
    merged = {d["id"]: d for d in existing if "id" in d}

    for d in new_dialogs:
        if d["id"] not in existing_ids:
            merged[d["id"]] = d

    merged_list = list(merged.values())
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_list, f, ensure_ascii=False, indent=2)

    print(f"\nБыло: {len(existing)} → Стало: {len(merged_list)} (+{len(merged_list) - len(existing)})")
    print(f"Файл сохранён: {output_path.resolve()}")


if __name__ == "__main__":
    print("=== Сбор реальных данных для Mental Health AI ===\n")

    all_dialogs = []

    print("[1/2] Парсинг открытых форумов...")
    all_dialogs.extend(scrape_b17_forum())
    all_dialogs.extend(scrape_psychology_forums())

    print(f"\n[2/2] Генерация кризисных диалогов...")
    all_dialogs.extend(generate_crisis_dialogues())

    if all_dialogs:
        save_and_merge(all_dialogs)
        print(f"\nСобрано диалогов: {len(all_dialogs)}")
    else:
        print("Не удалось собрать данные. Проверьте интернет.")


__all__ = [
    "scrape_b17_forum",
    "scrape_psychology_forums",
    "generate_crisis_dialogues",
    "save_and_merge",
    "anonymize",
]
