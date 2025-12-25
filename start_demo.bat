@echo off
cd /d "C:\Users\Yash Nautiyal\Desktop\Hand Control\hamoco"
call ".\.venv\Scripts\activate.bat"
.\.venv\Scripts\hamoco-run.exe --show --sensitivity 0.5
pause
