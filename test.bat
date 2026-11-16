@echo off
title ChurnIQ - Test Suite
set "PY_EXE=C:\Users\theju\AppData\Local\Programs\Python\Python311\python.exe"
if exist "%PY_EXE%" (
    "%PY_EXE%" -m pytest -v
) else (
    pytest -v
)
pause
