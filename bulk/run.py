# -*- coding: utf-8 -*-

import csv
import time
from pytube import YouTube
import os
from os.path import exists
import sys
sys.path.insert(0, os.path.dirname(__file__))
import multiprocessing
sys.path.insert(1, './backend')
import config
from extractor import SubtitleExtractor
import cv2

folderWrk = './bulk'
folderDownload = folderWrk + '/download'

fileList = folderWrk + '/list.csv'
fileDone = folderWrk + '/done.csv'
fileVideoPath = folderWrk + '/video_path.tmp'

hasMargin = True
marginPercentage = 1
marginSize = 10

def searchFile(file_path, word):
    with open(file_path, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            # check if string present on a current line
            if line.find(word) != -1:
                return lines.index(line) + 1
    return False
  
def extractSubs():
    if not exists(fileVideoPath):
      print('No video path found in: ' + fileVideoPath)
      exit()
      
    multiprocessing.set_start_method("spawn")
    with open(fileVideoPath, 'r') as file:
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

def processList(inputList):
    with open(inputList, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            print('********************************')
            ytId = row[0]
            lang = row[1]
            
            lineNum = searchFile(fileDone, ytId)
            
            if lineNum != False:
                print('Processed earlier, found YT: %s, in Done list at Line No. %s' % (ytId, lineNum))
                continue
            
            print('Processing YT: ' + ytId)

            fileSrt = folderDownload + '/' + ytId + '.srt'
            fileMp4 = folderDownload + '/' + ytId + '.mp4'
            if exists(fileMp4):
              print('File exists, no need to download: ' + fileMp4)
            else:
              print('Download file: ' + fileMp4)
              # YouTube('https://youtu.be/' + ytId).streams.first().download()
              yt = YouTube('https://youtube.com/watch?v=' + ytId)
              # resolucoes = yt.streams.all()
              # for i in resolucoes:  # mostra as resoluções disponíveis
              #   print(i)
              # exit()
        
              # yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download(folderDownload, ytId + '.mp4')
              yt.streams.filter(file_extension='mp4').order_by('resolution').desc().first().download(folderDownload, ytId + '.mp4')
              print('Download complete YT: %s to: %s' % (ytId, fileMp4))
              
              print('Waiting for 5 seconds')
              time.sleep(5)

            with open(fileVideoPath, 'w') as fd:
                fd.write(fileMp4)
            
            print('Written video path in: ' + fileVideoPath)
            
            if exists(fileSrt):
              print('File exists, no need to extract: ' + fileSrt)
              continue
            else:
              print('Extracting file: ' + fileSrt)
              extractSubs()
              print('Waiting for 5 seconds')
              time.sleep(5)
              
            if exists(fileSrt):
              print('Extraction complete: ' + fileSrt)
              os.remove(fileMp4)
              os.remove(fileVideoPath)
              with open(fileDone, 'a') as fd:
                  fd.write(ytId + "\n") 
            else:
              print('Extraction failed: ' + fileSrt)
              exit()
                      
                
            # exit()
            
if __name__ == '__main__':
  processList(fileList)