@echo off
rem Double-click to start the R-SAFE window app.
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\rsafe_app.py %*
if errorlevel 1 pause
