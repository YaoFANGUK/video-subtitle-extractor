cd /d %~dp0
CALL conda activate videoEnv
taskkill /f /im python*
taskkill /f /im VideoSubFinderWXW*
python backend\cli.py %*
