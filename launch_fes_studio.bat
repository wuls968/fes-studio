@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

if exist "%ROOT%\.venv\Scripts\python.exe" (
  set "PYTHON_BIN=%ROOT%\.venv\Scripts\python.exe"
  set "PYTHON_ARGS="
) else (
  where py >nul 2>nul
  if %errorlevel%==0 (
    set "PYTHON_BIN=py"
    set "PYTHON_ARGS=-3"
  ) else (
    set "PYTHON_BIN=python"
    set "PYTHON_ARGS="
  )
)

set "PYTHONPATH=%ROOT%src;%PYTHONPATH%"
call "%PYTHON_BIN%" %PYTHON_ARGS% -m fes_studio.launcher launch %*
endlocal
