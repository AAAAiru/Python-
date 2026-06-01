@echo off
REM Double-click to open the GUI (no need to keep this terminal open for each test).
cd /d "%~dp0"
if exist ".venv\Scripts\pythonw.exe" (
    start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0scripts\run_gui.py"
) else if exist ".venv\Scripts\python.exe" (
    start "" "%~dp0.venv\Scripts\python.exe" "%~dp0scripts\run_gui.py"
) else (
    start "" pythonw "%~dp0scripts\run_gui.py"
)
exit /b 0
