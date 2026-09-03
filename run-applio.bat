@echo off

if /i "%cd%"=="C:\Windows\System32" (
    color 0C
    echo Applio does not require administrator permissions and should be run as a regular user.
    echo.
    pause
    exit /b 1
)

setlocal
for %%F in ("%~dp0.") do set "folder_name=%%~nF"

title %folder_name%

where git >nul 2>nul
if not errorlevel 1 (
    echo Checking for updates...
    git -c http.sslBackend=openssl -C "%~dp0" pull --ff-only
    if errorlevel 1 echo Update failed. Starting the current version.
    echo.
)

if not exist env (
    echo Please run 'run-install.bat' first to set up the environment.
    pause
    exit /b 1
)

env\python.exe app.py --open
echo.
pause
