@echo off
REM Script to initialize git and connect to GitHub repository

echo Initializing git repository...
git init

echo Adding remote repository...
git remote add origin https://github.com/Thespaceblade/Brain-ML-Model.git

echo Checking remote...
git remote -v

echo Adding all files...
git add .

echo Creating initial commit...
git commit -m "Initial commit: Brain bleeding classification ML model"

echo.
echo Setup complete!
echo.
echo Next steps:
echo 1. Make sure you have git installed and configured
echo 2. Run: git push -u origin main
echo    (or git push -u origin master if your default branch is master)
echo.
pause



