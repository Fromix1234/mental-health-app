@echo off
chcp 65001 >nul
title Mental Health AI — Windows сборка
echo ============================================
echo  Mental Health AI — Windows сборка
echo ============================================
echo.

:: Проверка Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ОШИБКА] Python не найден!
    echo Скачайте: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Установка PyInstaller
echo [1/4] Устанавливаю PyInstaller...
pip install pyinstaller -q

:: Генерация датасета
echo [2/4] Генерирую датасет...
python -c "import sys; sys.path.insert(0, '.'); from data.dataset import generate_dataset; generate_dataset(50000)"
echo   -^> data/therapy_data.json готов

:: Сборка .exe
echo [3/4] Собираю Mental Health AI.exe...
pyinstaller --onefile --windowed --console ^
    --name "Mental Health AI" ^
    --add-data "data;data" ^
    --icon NONE ^
    --version-file NONE ^
    entry_web.py

:: Очистка
echo [4/4] Очищаю...
rmdir /s /q build 2>nul
del "Mental Health AI.spec" 2>nul

echo.
echo ============================================
echo  Готово!
echo ============================================
echo.
echo  Файл: dist\Mental Health AI.exe
echo.
echo  Просто открой его — откроется браузер
echo  Для выхода: нажми Ctrl+C в окне
echo.
pause
