@echo off
setlocal
cd /d "%~dp0"

if not exist ".\.venv-dagster\Scripts\python.exe" (
  echo Python Dagster introuvable. Cree .venv-dagster avec requirements-dagster.txt.
  exit /b 1
)

if not exist ".dagster" mkdir ".dagster"
if not exist ".dagster\dagster.yaml" copy /Y "dagster_pipeline\dagster.yaml" ".dagster\dagster.yaml" >nul
set "DAGSTER_HOME=%CD%\.dagster"
set "PYTHONUTF8=1"

.\.venv-dagster\Scripts\python.exe -m dagster_pipeline.sync_partitions
if errorlevel 1 exit /b %errorlevel%

echo Dagster disponible sur http://127.0.0.1:3000
.\.venv-dagster\Scripts\dagster.exe dev -m dagster_pipeline.definitions -h 127.0.0.1 -p 3000
