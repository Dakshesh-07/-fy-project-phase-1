@echo off
echo ============================================================
echo  Z24 Bridge SHM Digital Twin  -  Startup
echo ============================================================
echo.

echo [1/2] Starting FastAPI backend on http://127.0.0.1:8000 ...
start "Z24-Backend" cmd /k "cd /d "%~dp0" && python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload"

echo [2/2] Starting frontend via npx serve on http://localhost:3000 ...
start "Z24-Frontend" cmd /k "cd /d "%~dp0frontend" && npx serve -l 3000 ."

timeout /t 5 /nobreak >nul

echo.
echo  Backend API : http://127.0.0.1:8000
echo  Swagger docs: http://127.0.0.1:8000/docs
echo  Frontend UI : http://localhost:3000
echo.
echo Opening browser...
start "" http://localhost:3000

echo Press any key to close this window (servers keep running in their own windows).
pause >nul
