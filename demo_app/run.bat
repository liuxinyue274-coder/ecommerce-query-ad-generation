@echo off
REM V5 Ad Generation Visualization Dashboard - Windows launcher
REM Default assumes this script is in demo_app under the project root.
REM If not, set KS_PROJECT_ROOT before running this script.

setlocal

REM === Project root ===
if "%KS_PROJECT_ROOT%"=="" set KS_PROJECT_ROOT=%~dp0..

REM === DeepSeek API key (only needed for deepseek_chat provider) ===
REM set DEEPSEEK_API_KEY=YOUR_API_KEY_HERE
REM === Local SFT model path (only needed for sft_local provider) ===
REM set SFT_MODEL_PATH=C:\path\to\sft\model

cd /d "%~dp0"

echo Project root: %KS_PROJECT_ROOT%
echo Launching Streamlit...
streamlit run app.py
endlocal
