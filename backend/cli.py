# -*- coding: utf-8 -*-

from operator import truediv
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import config
import multiprocessing
from extractor import SubtitleExtractor
import cv2

currFolder = '/Users/sameer.shemna/Private/Test/Python/video-subtitle-extractor/backend'
# currFolder = '/home/ec2-user/work/video-subtitle-extractor/backend'

hasMargin = True
marginPercentage = 1
marginSize = 10

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    with open(currFolder + '/video_path.tmp', 'r') as file:
      video_path = file.read().rstrip()

    vid = cv2.VideoCapture(video_path)
    height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)

    y_min=0
    y_max=int(height)
    x_min=0
    x_max=int(width)
    subtitle_area = (y_min, y_max, x_min, x_max)
    
    if hasMargin and marginPercentage > 0:
      marginSizeY = int(y_max * marginPercentage / 100)
      marginSizeX = int(x_max * marginPercentage / 100)
      y_min = 0 + marginSizeY
      y_max = y_max - marginSizeY
      x_min = 0 + marginSizeX
      x_max = x_max - marginSizeX
      subtitle_area = (y_min, y_max, x_min, x_max)

    # (224,988,0,1920)
    # subtitle_area = (224, 988, 0, 1920)
    # subtitle_area = None

    print('Video path: ' + video_path)
    print('Area:\t\ty_min\ty_max\tx_min\tx_max')
    print('Values:\t\t' + str(y_min) + '\t' + str(y_max) + '\t' + str(x_min) + '\t' + str(x_max))

    se = SubtitleExtractor(video_path, subtitle_area)
    se.run()
