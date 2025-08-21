@echo off
setlocal

:: Navigate to project directory
cd /d "%~dp0"

echo Starting Varun AI Agent...
echo Project directory: %CD%

:: Check if ollama serve is running
tasklist /fi "imagename eq ollama.exe" | find /i "ollama.exe" >nul
if errorlevel 1 (
    echo Starting ollama serve...
    start /b ollama serve
    timeout /t 3 /nobreak >nul
) else (
    echo Ollama is already running
)

:: Wait for ollama to be ready with timeout
echo Waiting for Ollama to be ready...
set /a counter=0
set /a timeout=60

:wait_loop
curl -s http://127.0.0.1:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 goto ollama_ready

set /a counter+=1
if %counter% geq %timeout% (
    echo ERROR: Ollama failed to start within %timeout% seconds
    pause
    exit /b 1
)

timeout /t 1 /nobreak >nul
if %counter% geq 10 (
    set /a remainder=%counter% %% 10
    if %remainder% equ 0 echo Still waiting for Ollama... (%counter%/%timeout% seconds)
)
goto wait_loop

:ollama_ready
echo Ollama is ready!

:: Check if model exists, pull if needed
ollama list | findstr /i "llama3.2:3b-instruct-q4_K_M" >nul
if errorlevel 1 (
    echo Pulling llama3.2:3b-instruct-q4_K_M model...
    ollama pull llama3.2:3b-instruct-q4_K_M
)

:: Start Gunicorn server
echo Starting Gunicorn server on port 8080...
gunicorn -b 0.0.0.0:8080 --timeout 300 --workers 1 scripts.server:app