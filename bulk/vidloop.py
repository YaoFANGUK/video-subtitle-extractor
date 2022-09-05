import csv
from os.path import exists
from pytube import YouTube

folderWrk = './bulk'
folderDownload = folderWrk + '/download'

fileList = folderWrk + '/list.csv'
fileDone = folderWrk + '/done.csv'

def searchFile(file_path, word):
    with open(file_path, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            # check if string present on a current line
            if line.find(word) != -1:
                return lines.index(line) + 1
    return False

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
      
            with open(fileDone, 'a') as fd:
                fd.write(ytId + "\n")        
                
            # exit()
            
if __name__ == '__main__':
  processList(fileList)