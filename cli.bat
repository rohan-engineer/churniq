@echo off
title ChurnIQ - CLI Prediction
set "PY_EXE=C:\Users\theju\AppData\Local\Programs\Python\Python311\python.exe"
if exist "%PY_EXE%" (
    "%PY_EXE%" src\cli_predict.py %*
) else (
    python src\cli_predict.py %*
)
pause
