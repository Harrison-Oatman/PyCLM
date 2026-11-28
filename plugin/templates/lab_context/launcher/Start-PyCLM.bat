@echo off
rem Start a Claude Code session with PyCLM as one of this microscope's users.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" %*
if errorlevel 1 pause
