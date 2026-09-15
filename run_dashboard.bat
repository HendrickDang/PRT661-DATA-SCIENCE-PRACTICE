@echo off
REM =========================================================================
REM Launcher for NT Crime Forecasting & Resource Planning Streamlit Dashboard
REM Uses the 'ai' conda environment (C:\conda_envs\ai)
REM =========================================================================

echo Starting NT Crime & Assault Forecasting Streamlit Dashboard...
echo Using Python environment: C:\conda_envs\ai

set "STREAMLIT_EXE=C:\conda_envs\ai\Scripts\streamlit.exe"

if exist "%STREAMLIT_EXE%" (
    "%STREAMLIT_EXE%" run app.py
) else (
    echo Streamlit not found at %STREAMLIT_EXE%. Trying conda run...
    conda run -n ai streamlit run app.py
)

pause

