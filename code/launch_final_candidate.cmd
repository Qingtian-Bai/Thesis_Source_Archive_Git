@echo off
chcp 65001 >nul
cd /d "%~dp0"
"D:\python 3.13.7\python.exe" run_final_candidate.py --source 0
pause
