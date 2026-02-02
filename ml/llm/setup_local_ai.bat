@echo off
echo ==========================================
echo  StockSense Local AI Setup
echo ==========================================
echo.

REM Check if Ollama is installed
where ollama >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] Ollama not found
    echo.
    echo Please install Ollama first:
    echo 1. Download from: https://ollama.com/download/windows
    echo 2. Run the installer
    echo 3. Restart this script
    echo.
    echo Or install via winget:
    echo    winget install Ollama.Ollama
    echo.
    pause
    exit /b 1
)

echo [+] Ollama found!
ollama --version
echo.

REM Check if model exists
echo Checking for qwen2.5:7b model...
ollama list | findstr /C:"qwen2.5:7b" >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] Model not found. Downloading qwen2.5:7b...
    echo This will download ~4.7GB and take 5-10 minutes.
    echo.
    echo [*] Preventing computer sleep during download...
    echo.
    
    REM Prevent sleep while downloading using powercfg
    powercfg /change standby-timeout-ac 0
    powercfg /change standby-timeout-dc 0
    
    ollama pull qwen2.5:7b
    
    REM Restore sleep settings
    powercfg /change standby-timeout-ac 30
    powercfg /change standby-timeout-dc 15
    
    REM Verify the model was actually downloaded
    ollama list | findstr /C:"qwen2.5:7b" >nul 2>nul
    if %errorlevel% neq 0 (
        echo [!] Download failed. You can retry - Ollama will resume from where it stopped.
        pause
        exit /b 1
    )
    
    echo.
    echo [+] Download complete! Sleep settings restored.
) else (
    echo [+] Model already installed!
)

echo.
echo ==========================================
echo  Testing LLM Integration
echo ==========================================
echo.

cd /d "%~dp0"
python test_llm.py

echo.
echo ==========================================
echo  Setup Complete!
echo ==========================================
echo.
echo You can now:
echo 1. Use the chatbot: POST http://localhost:8000/llm/chat
echo 2. Generate scenarios: POST http://localhost:8000/llm/generate-scenario
echo 3. Check status: GET http://localhost:8000/llm/status
echo.
echo See README.md for more details.
echo.
pause
