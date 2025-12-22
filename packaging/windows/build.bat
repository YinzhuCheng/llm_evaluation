@echo off
setlocal enabledelayedexpansion

REM Build Windows GUI executable (folder mode).
REM Run this from repo root.

cd /d "%~dp0..\.."

if not exist ".venv" (
  echo Creating venv...
  py -3 -m venv .venv
)

call ".venv\Scripts\activate.bat"

echo Installing dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo Cleaning previous builds...
if exist "build" rmdir /s /q "build"
if exist "dist" rmdir /s /q "dist"

echo Building with PyInstaller...
python -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --name "EvalTool" ^
  --windowed ^
  --collect-all "plotly" ^
  --collect-submodules "plotly" ^
  --hidden-import "plotly.express" ^
  --hidden-import "plotly.graph_objects" ^
  --hidden-import "eval_questions" ^
  "gui_param_tool.py"

echo.
echo Build complete.
echo Output: dist\EvalTool\
echo.
pause

