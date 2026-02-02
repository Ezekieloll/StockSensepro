@echo off
REM Safe download script that prevents computer sleep
REM Usage: download_model.bat [model_name]

if "%1"=="" (
    set MODEL=qwen2.5:14b
) else (
    set MODEL=%1
)

echo ==========================================
echo  Downloading Ollama Model: %MODEL%
echo ==========================================
echo.
echo [*] Preventing computer sleep during download...
echo     Screen can turn off, but computer will stay awake
echo.

REM Temporarily disable sleep
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0
powercfg /change hibernate-timeout-ac 0
powercfg /change hibernate-timeout-dc 0

echo [*] Starting download...
ollama pull %MODEL%

REM Restore sleep settings
echo.
echo [*] Restoring power settings...
powercfg /change standby-timeout-ac 30
powercfg /change standby-timeout-dc 15
powercfg /change hibernate-timeout-ac 0
powercfg /change hibernate-timeout-dc 0

echo.
if %errorlevel% equ 0 (
    echo [+] Download complete!
) else (
    echo [!] Download interrupted. Run this script again to resume.
)
echo.
pause
