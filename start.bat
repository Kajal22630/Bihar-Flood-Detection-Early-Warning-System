@echo off
title Bihar Flood Prediction Dashboard
echo ============================================================
echo  BIHAR FLOOD PREDICTION DASHBOARD
echo  Starting server...
echo ============================================================
echo.
echo  [STEP 1] Activating virtual environment...
call venv\Scripts\activate.bat
echo.
echo  [STEP 2] Starting Flask server (http://127.0.0.1:5000)...
echo  Press CTRL+C to stop the server.
echo.
python app.py
pause
