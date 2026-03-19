@echo off
setlocal

set "ROOT=%~dp0"
cd /d "%ROOT%"

where py >nul 2>nul
if %errorlevel%==0 (
  set "PYTHON_BIN=py"
  set "PYTHON_ARGS=-3"
) else (
  set "PYTHON_BIN=python"
  set "PYTHON_ARGS="
)

call "%PYTHON_BIN%" %PYTHON_ARGS% -m venv .venv
if errorlevel 1 exit /b 1

call "%ROOT%\.venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

call "%ROOT%\.venv\Scripts\pip.exe" install -e .
if errorlevel 1 exit /b 1

echo.
echo FES Studio has been installed into %ROOT%\.venv
echo Launch it with:
echo   launch_fes_studio.bat
endlocal
