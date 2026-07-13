@echo off
setlocal
cd /d "%~dp0"

if not exist ".\.venv-prefect\Scripts\python.exe" (
  echo Python Prefect introuvable. Cree .venv-prefect avec requirements-prefect.txt.
  exit /b 1
)

if not exist ".\.venv\Scripts\python.exe" (
  echo Python GPU du pipeline introuvable: .\.venv\Scripts\python.exe
  exit /b 1
)

docker compose up -d postgres prefect
set "PREFECT_API_URL=http://127.0.0.1:4200/api"
set "PIPELINE_PYTHON=%CD%\.venv\Scripts\python.exe"
set "PYTHONUTF8=1"

.\.venv-prefect\Scripts\python.exe scripts\init\RUN_PIPELINE_PREFECT.py %*
