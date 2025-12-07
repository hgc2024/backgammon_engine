@echo off
echo Creating virtual environment '.venv' with Python 3.10...
py -3.10 -m venv .venv

if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
    
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
    
    echo.
    echo Setup complete!
    echo To activate the environment in the future, run:
    echo .venv\Scripts\activate
) else (
    echo Error: Failed to create virtual environment.
    exit /b 1
)
