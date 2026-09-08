@echo off
chcp 65001 >nul
cd /d "%~dp0"
env\python.exe realtime_gui.py
pause
