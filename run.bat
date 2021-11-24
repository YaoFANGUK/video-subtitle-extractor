cd /d %~dp0 
CALL conda activate videoEnv
python gui.py %1
rem pause