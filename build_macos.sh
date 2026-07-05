#!/bin/bash
set -e

echo "=== Mental Health AI — macOS сборка ==="

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "Установи Python 3: https://www.python.org/downloads/"
    exit 1
fi

# Установка зависимостей
echo "[1/4] Устанавливаю PyInstaller..."
pip3 install pyinstaller --quiet

# Генерация датасета
echo "[2/4] Генерирую датасет..."
python3 -c "import sys; sys.path.insert(0, '.'); from data.dataset import generate_dataset; generate_dataset(50000)"
echo "  -> data/therapy_data.json готов"

# Сборка .app
echo "[3/4] Собираю Mental Health AI.app..."
pyinstaller --onefile --windowed \
    --name "Mental Health AI" \
    --add-data "data:data" \
    --osx-bundle-identifier com.mentalhealth.app \
    entry_web.py

# Чистим мусор
echo "[4/4] Очищаю..."
rm -rf build entry_web.spec

echo ""
echo "============================================"
echo " Готово!"
echo "============================================"
echo ""
echo "Приложение: dist/Mental Health AI.app"
echo ""
echo "Просто открой его — откроется браузер"
echo "Для выхода: нажми Ctrl+C в терминале"
echo ""
