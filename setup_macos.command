#!/bin/bash
cd "$(dirname "$0")"

echo "==========================================="
echo " Mental Health AI"
echo " Быстрый запуск без установки Python"
echo "==========================================="
echo ""

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "[1/2] Устанавливаю Python..."
    if command -v brew &> /dev/null; then
        brew install python@3.11
    else
        echo "Открой https://www.python.org/downloads/"
        echo "Скачай и установи Python 3.11+"
        echo "Затем запусти этот файл снова"
        read -p "Нажми Enter для выхода..."
        exit 1
    fi
fi

echo "[1/2] Python OK ✓"

echo "[2/2] Запускаю приложение..."
echo ""
echo "Откроется браузер с интерфейсом"
echo "Для выхода закрой это окно (Ctrl+C)"
echo ""

python3 web_interface.py

echo ""
read -p "Нажми Enter для выхода..."
