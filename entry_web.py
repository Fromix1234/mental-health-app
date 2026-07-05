#!/usr/bin/env python3
"""Точка входа для PyInstaller сборок (macOS .app / Windows .exe)"""
import sys
import os
import webbrowser

if getattr(sys, 'frozen', False):
    os.chdir(os.path.dirname(sys.executable))

from web_interface import main

if __name__ == "__main__":
    webbrowser.open("http://localhost:8765")
    main()
