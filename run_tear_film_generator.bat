@echo off
REM Cross-platform batch file for Windows to run the tear film generator
REM This uses relative paths and should work from any Python installation

echo === Tear Film Theoretical Spectra Generator ===
echo.

REM Change to script directory
cd /d "%~dp0"

REM Run the Python runner script (pass through args, e.g., --ui)
python run_tear_film_generator.py %*

REM Pause to see results
pause
