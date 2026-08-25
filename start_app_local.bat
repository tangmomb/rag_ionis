@echo off
setlocal
cd /d "%~dp0"

set "ENV_FILE=.env.local"
if not exist "%ENV_FILE%" (
  echo Fichier .env.local introuvable. Cree-le depuis .env.local.example ou copie ton .env local.
  exit /b 1
)
set "RAG_IONIS_ENV=local"

for /f "usebackq tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
  if /i "%%A"=="OPENAI_API_KEY" set "OPENAI_API_KEY=%%B"
  if /i "%%A"=="S3_BUCKET_NAME" set "S3_BUCKET_NAME=%%B"
)

set "API_PYTHON=.\.venv\Scripts\python.exe"

if not exist "%API_PYTHON%" (
  echo Python du venv introuvable. Cree .venv avec requirements.txt.
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$patterns = @('uvicorn interface.app:app','uvicorn utils.app_database_browser.app:app','uvicorn utils.app_llm_tester.app:app','uvicorn utils.app_phoenix_replay.app:app','http.server 8003','watchfiles'); Get-CimInstance Win32_Process | Where-Object { $process = $_; $process.Name -eq 'python.exe' -and ($patterns | Where-Object { $process.CommandLine -like ('*' + $_ + '*') }) } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

docker compose --env-file "%ENV_FILE%" up -d
if errorlevel 1 (
  echo Les services Docker locaux n'ont pas pu demarrer.
  exit /b 1
)

start "RAG IONIS API" cmd /k "%API_PYTHON% -m uvicorn interface.app:app --host 127.0.0.1 --port 8006 --reload --reload-dir interface"
start "Database browser" cmd /k "%API_PYTHON% -m uvicorn utils.app_database_browser.app:app --host 127.0.0.1 --port 8001 --reload --reload-dir utils/app_database_browser"
start "LLM tester" cmd /k "%API_PYTHON% -m uvicorn utils.app_llm_tester.app:app --host 127.0.0.1 --port 8002 --reload --reload-dir utils/app_llm_tester --reload-dir interface/backend"
start "Phoenix Replay" cmd /k "%API_PYTHON% -m uvicorn utils.app_phoenix_replay.app:app --host 127.0.0.1 --port 8004 --reload --reload-dir utils --reload-dir interface"
start "Apps statiques" cmd /k "%API_PYTHON% -m http.server 8003 --bind 127.0.0.1 --directory utils"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ready = $false; 1..40 | ForEach-Object { try { $client = [Net.Sockets.TcpClient]::new(); $client.Connect('127.0.0.1', 8003); $client.Dispose(); $ready = $true; break } catch { Start-Sleep -Milliseconds 250 } }; if (-not $ready) { exit 1 }"
if errorlevel 1 (
  echo Le portail des applications n'a pas repondu sur le port 8003.
  exit /b 1
)

start "" "http://127.0.0.1:8003/app_launcher/"
