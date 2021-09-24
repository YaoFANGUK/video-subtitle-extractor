# -*- coding: UTF-8 -*-  
"""
@author: Fang Yao
@file  : settings.py
@time  : 2021/09/10 09:20
@desc  : 
"""
import PySimpleGUI as sg


def set_language_mode(config_file):
    languages = ['中文', '日语', '韩语', '法语', '德语']
    modes = ['[慢速] 高精准时间轴', '[快速] 低精准时间轴']
    window = sg.Window(title='字幕提取器',
                       layout=[
                           [sg.Text('选择视频字幕的语言:'), sg.DropDown(values=languages, size=(30, 20), pad=(0, 20),
                                                               key='-LANGUAGE-', readonly=True)],
                           [sg.Text('选择识别模式:'), sg.DropDown(values=modes, size=(30, 20), pad=(0, 20),
                                                               key='-MODE-', readonly=True)],
                           [sg.OK(), sg.Cancel()]
                       ])
    event, values = window.read()
    if event == 'OK':
        # 设置模型语言配置
        print('选择了:', values['-LANGUAGE-'])
        language = None
        mode = None
        if values['-LANGUAGE-'] == '中文':
            # 设置中文模型
            language = 'ch'
        elif values['-LANGUAGE-'] == '日语':
            # 设置日语模型
            language = 'japan'
        elif values['-LANGUAGE-'] == '韩语':
            # 设置韩语模型
            language = 'korean'
        elif values['-LANGUAGE-'] == '法语':
            # 设置法语模型
            language = 'french'
        elif values['-LANGUAGE-'] == '德语':
            # 设置法语模型
            language = 'german'
        # 设置模型语言配置
        print('选择了:', values['-MODE-'])
        if values['-MODE-'] == '[慢速] 高精准时间轴':
            # 设置精准模式
            mode = 'accurate'
        elif values['-MODE-'] == '[快速] 低精准时间轴':
            # 设置快速模式
            mode = 'fast'
        set_config(config_file, language, mode)
        window.close()


def set_config(config_file, language_code, mode):
    # 写入配置文件
    with open(config_file, mode='w') as f:
        f.write('[DEFAULT]\n')
        f.write(f'Language = {language_code}\n')
        f.write(f'Mode = {mode}\n')


