#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import re
import time
import sys
import os


codings = ["utf-8", "utf-16", "utf-32", "gbk", "gb2312", "gb18030", "big5", "cp1252"]


def readings(filename='', content='', encoding=''):
    """
    获取文件内容
    filename: 文件名
    content: 二进制或者字符串
    encoding: 文件编码
    """

    # 传入的内容不是字符串
    if not content:
        if not encoding:
            for coding in codings:
                try:
                    with open(filename, 'r', encoding=coding) as file:
                        content = file.read()
                    break
                except (UnicodeDecodeError, UnicodeError):
                    print(f"使用{coding}解码失败,尝试替换编码格式.")
                    continue
                except FileNotFoundError:
                    print("未能找到指定文件,程序即将退出.")
                    time.sleep(3)
                    sys.exit(1)
        else:
            with open(filename, 'r', encoding=encoding) as file:
                content = file.read()
    return content


def srt_timestamps(content):
    """ 获取时间轴存于字典中 """
    timestamps = []
    for ts in re.findall(r'\d{2}:\d{2}:\d{2},\d{3}.+\d{2}:\d{2}:\d{2},\d{3}', content):
        ts = ts.split(' --> ')
        timestamps.append(ts)
    return timestamps


def srt_subtitles(content):
    # 通过分割时间轴获取字幕存储于列表中返回
    content = content.replace('\ufeff', '')
    _subtitles = re.split(r'\d{2}:\d{2}:\d{2},\d{3}.+\d{2}:\d{2}:\d{2},\d{3}', content)
    subtitles = []
    for s in range(1, len(_subtitles)):
        # print(re.sub('(\r\n)|\n', '', _subtitles[s]))
        subtitle = re.sub(r'(\r\n\r\n\d+\r\n)|(^\r\n)|(^\n)|(\n\n\d+\n)|(\u200e)', '', _subtitles[s])
        subtitle = re.sub('(\r\n)|\n', '|', subtitle.strip())
        subtitle = subtitle.replace("||", '\\N{\\r原文字幕}')
        subtitles.append(subtitle)

        # 
    return subtitles


def ass_content(timestamps, subtitles, header_path):
    content = readings(filename=header_path) + '\n'
    body = {
        'dialogue': 'Dialogue: ',
        'front_time': '',
        'behind_time': '',
        'default': 'Default',
        'ntp': 'NTP',
        '0000': '0000',
        'sub': ',',
    }
    count = len(subtitles)
    for c in range(count):
        start = timestamps[c][0]                    # 获取当前字幕起始时间
        start = start[:1] + ',' + start[1:8] + '.' + start[-2:]
        end = timestamps[c][1]                      # 获取当前字幕结束时间
        end = end[1:8] + '.' + end[-2:]
        timeline = ','.join([start, end])           # 合成时间轴

        subtitle = subtitles[c]                     # 获取当前字幕

        list2str = [  # 字幕列表格式化
            body['dialogue'] + timeline,
            body['default'],
            body['ntp'],
            body['0000'],
            body['0000'],
            body['0000'] + ',',
            subtitle]

        content += ','.join(list2str)
        content += '\n'

    return content


def srt_to_ass(filename='', content='', **kwargs):
    start_time = time.time()
    for_json = kwargs.get('for_json')
    header_path = kwargs.get('header_path') if kwargs.get('header_path') else './header.txt'
    encoding = kwargs.get('encoding')
    content = readings(filename=filename, content=content, encoding=encoding)
    timestamps = srt_timestamps(content)
    subtitles = srt_subtitles(content)
    content = ass_content(timestamps, subtitles, header_path)
    end_time = time.time() - start_time
    if for_json:
        data = {
            'time': end_time,
            'content': content
        }
        return data
    return content

def write_srt_to_ass(filename):
    filename_ass = re.sub(r'\.srt$', '.ass', filename)
    with open(filename_ass, 'w', encoding='utf-8') as to_ass:
        to_ass.write(srt_to_ass(filename))
    print(filename_ass, '[finish]')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv)):
            filename = sys.argv[i]
            write_srt_to_ass(filename)
    else:
        print('扫描当前目录下的srt文件...')
        lis = [li for li in os.listdir() if re.search(r'\.srt$', li)]
        for li in lis:
            print(li)
        i = input('是否将上述文件转换为ass文件(yes/no):')
        while i not in ('yes', 'no'):
            i = input('是否将上述文件转换为ass文件(yes/no):')
        if i == 'yes':
            for li in lis:
                filename = li
                filename_ass = re.sub(r'\.srt$', '.ass', filename)
                with open(filename_ass, 'w', encoding='utf8') as to_ass:
                    to_ass.write(srt_to_ass(filename))
                print("完成...")
        else:
            print("取消操作...")
        time.sleep(5)