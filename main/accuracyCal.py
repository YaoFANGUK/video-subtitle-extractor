# coding:utf-8
import glob
import difflib
import codecs
import re
import webbrowser
import os
import Levenshtein

def is_num_by_except(num):
    try:
        int(num)
        return True
    except ValueError:
#        print "%s ValueError" % num
        return False

# 去除非中英文字符
def cleantxt(raw):
    fil = re.compile('[^\u4e00-\u9fa5^.^a-z^A-Z^]', re.UNICODE)
    return fil.sub('', raw)

def delete_timeline_and_no():
    for i in range(3):
        for line1 in srt1:
            if is_num_by_except(cleantxt(line1)) :
                srt1.remove(line1)
            else:
                if '-->' in line1:
                    srt1.remove(line1)
                else:
                    if (cleantxt(line1) == ''):
                        srt1.remove(line1)

        for line2 in srt2:
            if is_num_by_except(cleantxt(line2)) :
                srt2.remove(line2)
            else:
                if '-->' in line2:
                    srt2.remove(line2)
                else:
                    if (cleantxt(line2) == ''):
                        srt2.remove(line2)

def calculate_accuracy(srt1, srt2):
    distance = Levenshtein.distance("".join(srt1),"".join(srt2))
    print('The Accuracy Rate is: ',(len(str(srt1)) - distance)/len(str(srt1)))
    print('total characters: ',len(str(srt1)))

if __name__ == '__main__':
    
    srt_original = input('Please Input original srt file: ').strip()
    srt_extracted = input('Please Input extracted srt file: ').strip()
    srt1 = ""
    srt2 = ""
    diff = difflib.HtmlDiff()
    with open(srt_original,'r') as f:
        srt1 = f.readlines()
    with open(srt_extracted, 'r') as f:
        srt2 = f.readlines()

    # with open(srt_path + 'difference.html','a+') as fo:
    #     fo.write(diff.make_file(srt1, srt2))

    delete_timeline_and_no()

    srt_path = os.getcwd() 
    with open(srt_path + '/data/difference.html','w') as fo:
        fo.write(diff.make_file(srt1, srt2))

    print('Please Open file \'' + srt_path + 'difference.html' + '\' to check the number of errors')
    calculate_accuracy(srt1,srt2)
