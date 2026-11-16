@echo off
title ChurnIQ - Platform Server
echo ========================================================
echo               Starting ChurnIQ Server...
echo ========================================================
echo URL: http://localhost:5000/
echo.

set "PY_EXE=C:\Users\theju\AppData\Local\Programs\Python\Python311\python.exe"
if exist "%PY_EXE%" (
    "%PY_EXE%" app.py
) else (
    python app.py
)
pause
