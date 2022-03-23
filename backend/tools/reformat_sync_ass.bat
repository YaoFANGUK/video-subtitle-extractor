cd /d %~dp0 
CALL conda activate videoEnv
python reformat_sync_ass.py %*
pause