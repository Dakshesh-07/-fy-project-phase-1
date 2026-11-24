@echo off
echo ============================================================
echo  Pushing Z24 Bridge Digital Twin to GitHub
echo  Repo: https://github.com/meo-ach/-fy-project-phase-1
echo ============================================================
echo.
cd /d "%~dp0"

echo Current Git status:
git status
echo.
echo Pushing to origin main (sign in via browser if prompted)...
git push -u origin main
echo.
if %ERRORLEVEL% equ 0 (
    echo ============================================================
    echo  SUCCESS! All files and folders pushed to GitHub.
    echo ============================================================
) else (
    echo ============================================================
    echo  Push encountered an issue. See message above.
    echo ============================================================
)
echo.
pause
