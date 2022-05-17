cd /d %~dp0
CALL conda activate Sushi
python run.py %*
pause