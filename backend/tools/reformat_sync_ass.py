# -*- coding: UTF-8 -*-
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import re
import shutil
from pathlib import Path
import zipfile

from reformat_en import reformat
from srt2ass import write_srt_to_ass
from translation import chs_to_cht
from sushi.sushi_main import subtitle_sync

def main(argv):
    videos = []
    subtitle = None
    if len(argv) < 2:
        print("参数不足, 程序退出")
        exit()
    for arg in argv:
        if re.search('\\.(srt)$', arg, flags=re.I) is not None:
            subtitle = arg
        elif re.search('\\.(srt)\\.bak$', arg, flags=re.I) is not None:
            subtitle = arg
        else:
            videos.append(arg)
    print(("字幕文件:", subtitle))
    if subtitle is None:
        print("未找到字幕文件, 程序退出")
        exit()
    if len(videos) < 2:
        print("未找到视频文件, 程序退出")
        exit()

    if os.path.getsize(videos[0]) > os.path.getsize(videos[1]):
        bdVideo = videos[0]
        hdVideo = videos[1]
    else:
        bdVideo = videos[1]
        hdVideo = videos[0]
    print(("HD:", hdVideo))
    print(("BD:", bdVideo))
    subtitle_new = os.path.splitext(subtitle)[0] + '.srt'
    subtitle_new = subtitle_new.replace(".srt.srt", ".srt")
    shutil.copyfile(subtitle, subtitle_new)
    subtitle_new = re.sub('\\.zh.+\\.srt', '.srt', subtitle_new)

    srt_filename = os.path.join(os.path.splitext(subtitle_new)[0] + '.zh.简体-英文.srt')
    srt_filename_cht = os.path.join(os.path.splitext(subtitle_new)[0] + '.zh.繁体-英文.srt')
    reformat(srt_filename, bdVideo)
    chs_to_cht(srt_filename, srt_filename_cht)
    write_srt_to_ass(srt_filename_cht)
    write_srt_to_ass(srt_filename)

    print("开始同步时间轴")
    temp_dir = os.path.join(os.path.dirname(__file__), '../../output/sushi')
    subtitle_sync([bdVideo, hdVideo, srt_filename], {'temp': temp_dir})
    bd_video_path_prefix = os.path.splitext(bdVideo)[0]
    srt_filename_eng = bd_video_path_prefix + '.eng.srt'
    srt_filename = bd_video_path_prefix + '.zh.简体-英文.srt'
    srt_filename_cht = bd_video_path_prefix + '.zh.繁体-英文.srt'
    ass_filename = bd_video_path_prefix + '.zh.简体-英文.ass'
    ass_filename_cht = bd_video_path_prefix + '.zh.繁体-英文.ass'
    os.replace(os.path.join(os.path.splitext(bdVideo)[0] + '.srt'), srt_filename)
    chs_to_cht(srt_filename, srt_filename_cht)
    write_srt_to_ass(srt_filename_cht, ass_filename_cht)
    write_srt_to_ass(srt_filename, ass_filename)
    # zip
    subtitle_path_list = [
        srt_filename_eng,
        srt_filename,
        srt_filename_cht,
        ass_filename,
        ass_filename_cht,
    ]
    with zipfile.ZipFile(bd_video_path_prefix + '.zh.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in subtitle_path_list:
            if os.path.exists(file):
                zf.write(file, Path(file).name)

    print("finish")

if __name__ == '__main__':
    main(sys.argv[1:])



