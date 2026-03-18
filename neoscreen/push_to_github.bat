@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM NeoScreen — GitHub Push Script (Windows)
REM Usage: push_to_github.bat YOUR_GITHUB_USERNAME
REM Example: push_to_github.bat rishav-jha
REM ─────────────────────────────────────────────────────────────────────────────

SET GITHUB_USER=%1
SET REPO_NAME=neoscreen

IF "%GITHUB_USER%"=="" (
    echo ERROR: Please provide your GitHub username.
    echo Usage: push_to_github.bat YOUR_GITHUB_USERNAME
    exit /b 1
)

SET REMOTE_URL=https://github.com/%GITHUB_USER%/%REPO_NAME%.git

echo ======================================================
echo  NeoScreen ^> GitHub Push
echo  Remote: %REMOTE_URL%
echo ======================================================

REM ── Create placeholder Flutter asset dirs ─────────────────────────────────
if not exist "flutter\assets\models" mkdir "flutter\assets\models"
if not exist "flutter\assets\data"   mkdir "flutter\assets\data"

REM ── Placeholder model (replaced after training) ───────────────────────────
if not exist "flutter\assets\models\neoscreen_v1.tflite" (
    echo. > "flutter\assets\models\neoscreen_v1.tflite"
    echo NOTE: Placeholder model created. Replace with real trained model after Day 1.
)

REM ── Seed PHC DB if missing ────────────────────────────────────────────────
if not exist "data\phc_maharashtra.db" (
    echo Seeding PHC database...
    python scripts\seed_phc_db.py
)

REM ── Copy DB to Flutter assets ─────────────────────────────────────────────
if exist "data\phc_maharashtra.db" (
    copy /Y "data\phc_maharashtra.db" "flutter\assets\data\phc_maharashtra.db" >nul
)

REM ── Git init if needed ────────────────────────────────────────────────────
if not exist ".git" (
    git init
    git branch -M main
)

REM ── Stage all files ───────────────────────────────────────────────────────
git add -A

REM ── Commit ────────────────────────────────────────────────────────────────
git commit -m "feat: NeoScreen initial codebase - IISc IdeaSprint 2026

- ml/: sclera detection, TF Lite inference, PHC referral, training, evaluation
- day1/: dataset setup, sclera test, model build, augmentation, 10-image test, misclassification log
- day2/: Flutter asset setup, PHC lookup test, Twilio test, end-to-end test
- flutter/: single-screen offline Android app (camera to result in one tap)
- NeoScreen_Training.ipynb: Google Colab T4 training notebook
- scripts/seed_phc_db.py: 100 Maharashtra PHC coordinates

Stack: MobileNetV3 TF Lite + Flutter + OpenCV + Twilio + SQLite
Target: Moto E13 (Rs 6000) - offline - Hindi/English - free"

REM ── Add remote if not already added ──────────────────────────────────────
git remote get-url origin >nul 2>&1
IF ERRORLEVEL 1 (
    git remote add origin %REMOTE_URL%
    echo Remote added: %REMOTE_URL%
) ELSE (
    git remote set-url origin %REMOTE_URL%
    echo Remote updated: %REMOTE_URL%
)

REM ── Push ─────────────────────────────────────────────────────────────────
echo.
echo Pushing to GitHub...
git push -u origin main

IF ERRORLEVEL 1 (
    echo.
    echo Push failed. Common fixes:
    echo   1. Make sure the repo exists on GitHub: https://github.com/%GITHUB_USER%/%REPO_NAME%
    echo   2. Check you are logged in: git config --global user.name "Your Name"
    echo   3. If using HTTPS, enter your GitHub password when prompted
    echo      ^(Use a Personal Access Token, not your GitHub password^)
    echo      Get one at: https://github.com/settings/tokens
) ELSE (
    echo.
    echo ======================================================
    echo  Success! NeoScreen is live on GitHub.
    echo  View at: https://github.com/%GITHUB_USER%/%REPO_NAME%
    echo ======================================================
)
