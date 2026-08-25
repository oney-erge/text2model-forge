@echo off
title Text2Model Forge
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
if errorlevel 1 (
  echo.
  echo Text2Model Forge did not start. Review the error above.
  pause
)
