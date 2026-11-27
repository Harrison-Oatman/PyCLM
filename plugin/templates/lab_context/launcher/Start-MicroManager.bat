@echo off
rem Start the Micro-Manager GUI the way this microscope needs it (launcher\local.json, "micromanager").
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0micromanager.ps1" %*
if errorlevel 1 pause
