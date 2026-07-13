@echo off
setlocal
cd /d "%~dp0"

if not exist ".\.venv-interface\Scripts\python.exe" (
  echo Python interface introuvable: .\.venv-interface\Scripts\python.exe
  exit /b 1
)

echo Explorateur disponible sur http://127.0.0.1:8003/videos
.\.venv-interface\Scripts\python.exe -m uvicorn utils.database_browser.app:app --host 127.0.0.1 --port 8003
