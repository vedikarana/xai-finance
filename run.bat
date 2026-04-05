@echo off
echo Starting StockSense AI...
echo.

REM Try Python 3.11 first
py -3.11 -m streamlit run app.py
if %ERRORLEVEL% NEQ 0 (
    echo Python 3.11 not found, trying venv...
    if exist venv311\Scripts\python.exe (
        venv311\Scripts\python.exe -m streamlit run app.py
    ) else (
        echo Creating virtual environment with Python 3.11...
        py -3.11 -m venv venv311
        venv311\Scripts\pip.exe install streamlit yfinance pandas numpy scikit-learn xgboost shap lime matplotlib seaborn plotly ta joblib scipy gtts
        venv311\Scripts\python.exe -m streamlit run app.py
    )
)
