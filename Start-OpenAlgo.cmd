@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\local-openalgo.ps1" -Action Start
if errorlevel 1 pause
