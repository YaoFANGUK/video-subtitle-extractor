currFolder='/Users/sameer.shemna/Private/Test/Python/video-subtitle-extractor/backend'
# currFolder='/home/ec2-user/work/video-subtitle-extractor/backend'

echo $1 > "$currFolder/video_path.tmp"

source activate videoEnv

python "$currFolder/cli.py"

conda deactivate
