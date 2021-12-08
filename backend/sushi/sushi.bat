cd /d %~dp0 
CALL conda activate Sushi
ffmpeg -i %1 %1.wav
ffmpeg -i %2 %2.wav
python sushi.py --src %1.wav --dst %2.wav --script %3
del /Q /F %1.wav
del /Q /F %2.wav
rem pause