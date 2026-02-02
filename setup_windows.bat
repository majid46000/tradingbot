@echo off
setlocal

set "PYTHON=python"
set "VENV=.venv"

echo ==> Checking Python...
%PYTHON% --version
if errorlevel 1 (
  echo ❌ Python not found. Install Python 3.12+ from https://www.python.org/downloads/
  exit /b 1
)

echo ==> Creating virtual environment at %VENV%...
%PYTHON% -m venv %VENV%
if errorlevel 1 (
  echo ❌ Failed to create virtual environment.
  exit /b 1
)

echo ==> Activating virtual environment...
call %VENV%\Scripts\activate.bat

echo ==> Upgrading pip...
python -m pip install --upgrade pip

echo ==> Installing dependencies...
pip install -r requirements.txt

echo ✅ Setup complete.
echo To activate later: %VENV%\Scripts\activate.bat
endlocal
