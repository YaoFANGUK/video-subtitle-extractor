# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao
@Time    : 2021/3/24 9:28 上午
@FileName: main.py
@desc: 主程序入口文件
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import config
import multiprocessing
from extractor import SubtitleExtractor

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    # 提示用户输入视频路径 Prompt the user to enter the video path
    video_path = input(f"{config.interface_config['Main']['InputVideo']}").strip()
    # 提示用户输入字幕区域 Prompt the user to enter the caption area
    try:
        y_min, y_max, x_min, x_max = map(int, input(
            f"{config.interface_config['Main']['ChooseSubArea']} (ymin ymax xmin xmax)：").split())
        subtitle_area = (y_min, y_max, x_min, x_max)
    except ValueError as e:
        subtitle_area = None
    print('Video path: ' + video_path)
    # 新建字幕提取对象 New subtitle extraction object
    se = SubtitleExtractor(video_path, subtitle_area)
    # 开始提取字幕 Start extracting subtitles
    se.run()
