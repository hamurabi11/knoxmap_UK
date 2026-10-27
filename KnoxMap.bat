@echo off
rem Launch KnoxMap with the project's virtualenv, no console window.
rem First run? Setup creates the environment and fetches the map tools.
cd /d "%~dp0"
rem UTF-8 for every file Python reads and writes, whatever the PC's code page.
rem On a Korean or Japanese Windows the default one cannot hold all of KnoxMap's text.
set PYTHONUTF8=1
if not exist ".venv\Scripts\pythonw.exe" (
  call Setup.bat
  if errorlevel 1 exit /b 1
)
rem KnoxMap UK: "KnoxMap.bat --uk" starts with the UK mode panel on (web/uk.py).
if /i "%~1"=="--uk" set KNOXMAP_UK=1
start "" ".venv\Scripts\pythonw.exe" "knoxmap.py"
