#!/bin/bash
set -e

echo "============================================"
echo " Mental Health AI — macOS сборка"
echo "============================================"
echo ""

APP_NAME="Mental Health AI"
IDENTIFIER="com.mentalhealth.app"

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Установи Python 3: https://www.python.org/downloads/"
    exit 1
fi

# Установка PyInstaller
echo "[1/4] Устанавливаю PyInstaller..."
pip3 install pyinstaller --quiet

# Генерация датасета
echo "[2/4] Генерирую датасет..."
python3 -c "
import sys; sys.path.insert(0, '.')
from data.dataset import generate_dataset
generate_dataset(50000)
"
echo "  -> data/therapy_data.json готов"

# Сборка .app
echo "[3/4] Собираю $APP_NAME.app..."
pyinstaller --onefile --windowed \
    --name "$APP_NAME" \
    --add-data "data:data" \
    --osx-bundle-identifier "$IDENTIFIER" \
    entry_web.py

rm -rf build "$APP_NAME.spec"

# Создание .dmg
echo "[4/4] Создаю $APP_NAME.dmg..."
DMG_NAME="${APP_NAME// /_}.dmg"
DMG_DIR="dist"

if [ -f "$DMG_DIR/$DMG_NAME" ]; then
    rm "$DMG_DIR/$DMG_NAME"
fi

# Используем hdiutil для создания .dmg
hdiutil create -volname "$APP_NAME" \
    -srcfolder "$DMG_DIR/$APP_NAME.app" \
    -ov -format UDZO \
    "$DMG_DIR/$DMG_NAME" 2>/dev/null

echo ""
echo "============================================"
echo "  Готово!"
echo "============================================"
echo ""
echo "  .app: dist/$APP_NAME.app"
echo "  .dmg: dist/$DMG_NAME"
echo ""
echo "  Размер: $(du -sh dist/$DMG_NAME | cut -f1)"
echo ""
echo "  Просто открой .dmg и перетащи .app в"
echo "  папку Программы"
echo ""
