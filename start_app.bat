@echo off
setlocal
cd /d "%~dp0"

for /f "usebackq tokens=1,* delims==" %%A in (".env") do (
  if /i "%%A"=="OPENAI_API_KEY" set "OPENAI_API_KEY=%%B"
  if /i "%%A"=="S3_BUCKET_NAME" set "S3_BUCKET_NAME=%%B"
)

set "API_PYTHON=.\.venv\Scripts\python.exe"

if not exist "%API_PYTHON%" (
  echo Python du venv introuvable. Cree .venv avec requirements.txt.
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "$patterns = @('uvicorn interface.app:app','uvicorn utils.database_browser.app:app','http.server 8002','watchfiles'); Get-CimInstance Win32_Process | Where-Object { $process = $_; $process.Name -eq 'python.exe' -and ($patterns | Where-Object { $process.CommandLine -like ('*' + $_ + '*') }) } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"

docker compose up -d postgres phoenix

start "RAG IONIS API" cmd /k "%API_PYTHON% -m uvicorn interface.app:app --host 127.0.0.1 --port 8006 --reload --reload-dir interface"
start "" "http://127.0.0.1:8006/"

start "Database browser" cmd /k "%API_PYTHON% -m uvicorn utils.database_browser.app:app --host 127.0.0.1 --port 8001 --reload --reload-dir utils/database_browser"
start "" "http://127.0.0.1:8001/videos"
start "" "http://127.0.0.1:6006/"
if exist ".\.venv\Scripts\dagster.exe" (
  powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 3000 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
  if errorlevel 1 (
    if not exist ".dagster" mkdir ".dagster"
    if not exist ".dagster\dagster.yaml" copy /Y "dagster_pipeline\dagster.yaml" ".dagster\dagster.yaml" >nul
    set "DAGSTER_HOME=%CD%\.dagster"
    set "PYTHONUTF8=1"
    .\.venv\Scripts\python.exe -m dagster_pipeline.sync_partitions
    if errorlevel 1 exit /b 1
    start "Dagster" cmd /k .\.venv\Scripts\dagster.exe dev -m dagster_pipeline.definitions -h 127.0.0.1 -p 3000
  )
  start "" powershell -NoProfile -WindowStyle Hidden -Command "$deadline=(Get-Date).AddSeconds(45); do { try { Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:3000/' -TimeoutSec 2 | Out-Null; Start-Process 'http://127.0.0.1:3000/assets'; exit } catch { Start-Sleep -Milliseconds 750 } } while ((Get-Date) -lt $deadline)"
) else (
  echo Dagster non lance: installe requirements.txt dans .venv.
)
start "" "https://www.youtube.com/@IONIS-STM/videos"

start "" "https://console.cloud.google.com/apis/dashboard?project=youtube-api-484522&pageState=(%%22duration%%22:(%%22groupValue%%22:%%22P2D%%22,%%22customValue%%22:null))"
start "" "https://eu-west-3.console.aws.amazon.com/s3/buckets/rag-ionis-532523613357-eu-west-3-an?region=eu-west-3&tab=objects"
start "" "https://platform.openai.com/usage"
start "" "https://developers.openai.com/api/docs/pricing"
start "" "https://platform.openai.com/tokenizer"
