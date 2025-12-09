@echo off
echo Starting Backgammon Gen 3 App...

:: Start Backend
echo Launching Backend (FastAPI)...
start "Backgammon Backend" cmd /k "call .venv\Scripts\activate && python -m uvicorn src.api:app --reload"

:: Start Frontend
echo Launching Frontend (React)...
cd frontend
start "Backgammon Frontend" cmd /k "npm run dev"

echo Done. Check the new windows.
