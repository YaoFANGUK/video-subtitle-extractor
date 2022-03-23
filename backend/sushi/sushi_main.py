# -*- coding: UTF-8 -*-
import sys 
import os
import re
from pathlib import Path
from .__main__ import parse_args_and_run

def subtitle_sync(argv, opts=None):
    videos = []
    isAss = False
    if len(argv) < 3:
        print("参数不足, 程序退出")
        exit()
    for arg in argv:
        if re.search('\\.(ass)$', arg, flags=re.I) is not None:
            subtitle = arg
            isAss = True
        elif re.search('\\.(srt)$', arg, flags=re.I) is not None:
            subtitle = arg
        else:
            videos.append(arg)
    print(("字幕文件:", subtitle))
    if subtitle is None:
        print("未找到字幕文件, 程序退出")
        exit()

    if os.path.getsize(videos[0]) > os.path.getsize(videos[1]):
        bdVideo = videos[0]
        hdVideo = videos[1]
    else:
        bdVideo = videos[1]
        hdVideo = videos[0]
    print(("HD:", hdVideo))
    print(("BD:", bdVideo))

    temp_dir = None
    if opts is not None:
        if "temp" in opts:
            temp_dir = opts["temp"]

    if temp_dir is not None:
        hdWav = os.path.join(temp_dir, os.path.basename(hdVideo) + '.wav')
        bdWav = os.path.join(temp_dir, os.path.basename(bdVideo) + '.wav')
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
    else:
        hdWav = hdVideo + '.wav'
        bdWav = bdVideo + '.wav'

    os.system('ffmpeg -y -i "{0}" "{1}"'.format(hdVideo, hdWav))
    os.system('ffmpeg -y -i "{0}" "{1}"'.format(bdVideo, bdWav))

    (prefix, sep, suffix) = bdVideo.rpartition('.')

    if isAss:
        output = re.sub('\\.' + suffix + '$', ".ass", bdVideo, flags= re.I)
    else:
        output = re.sub('\\.' + suffix + '$', ".srt", bdVideo, flags= re.I)

    parse_args_and_run([
        "--src", hdWav,
        "--dst", bdWav,
        "--script", subtitle,
        "--output", output,
    ])

    os.remove(hdWav)
    os.remove(bdWav)
    print(("finish", output))

if __name__ == '__main__':
    subtitle_sync(sys.argv[1:], None)



