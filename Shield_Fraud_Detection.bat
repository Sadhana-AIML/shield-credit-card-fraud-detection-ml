@echo off
title Shield Credit Card Fraud Detection
cd /d "C:\Users\cusha\Downloads\Credit Card Fraud Detection Using ML\Credit Card Fraud Detection Using ML\backend"

echo ===============================================
echo   Shield Credit Card Fraud Detection
echo ===============================================
echo.
echo Starting backend server...
echo.

start "Shield Backend" cmd /k "cd /d C:\Users\cusha\Downloads\Credit Card Fraud Detection Using ML\Credit Card Fraud Detection Using ML\backend && python -u app.py"

timeout /t 3 /nobreak >nul

echo Opening frontend...
start "" "C:\Users\cusha\Downloads\Credit Card Fraud Detection Using ML\Credit Card Fraud Detection Using ML\frontend\index.html"

echo.
echo Project started.
echo Keep the backend window open while using the project.
pause
