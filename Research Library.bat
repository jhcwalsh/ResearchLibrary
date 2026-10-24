@echo off
rem Launches the Research Library app. Copy this file anywhere (e.g. the desktop); it always
rem runs from the repo folder below. Edit APP_DIR if the repo moves.
set "APP_DIR=C:\Users\james\PycharmProjects\ResearchLibrary"
title Research Library
cd /d "%APP_DIR%" || (echo Cannot find %APP_DIR% & pause & exit /b 1)
if not exist ".venv\Scripts\streamlit.exe" (
    echo No .venv\Scripts\streamlit.exe in %APP_DIR%. Create the venv and pip install -e .[dev] streamlit
    pause
    exit /b 1
)
echo Starting the Research Library app. Close this window to stop it.
".venv\Scripts\streamlit.exe" run app.py --server.headless false
if errorlevel 1 pause
