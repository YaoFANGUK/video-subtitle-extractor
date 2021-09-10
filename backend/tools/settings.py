# -*- coding: UTF-8 -*-  
"""
@author: Fang Yao
@file  : settings.py
@time  : 2021/09/10 09:20
@desc  : 
"""
import PySimpleGUI as sg


def set_language(config_file):
    languages = ['中文', '日语', '韩语', '法语', '德语']
    window = sg.Window(title='字幕提取器',
                       layout=[
                           [sg.Text('选择视频字幕的语言:'), sg.DropDown(values=languages, size=(30, 20), pad=(0, 20),
                                                               key='-LANGUAGE-')],
                           [sg.OK(), sg.Cancel()]
                       ])
    event, values = window.read()
    if event == 'OK':
        # 设置模型语言配置
        print('选择了:', values['-LANGUAGE-'])
        if values['-LANGUAGE-'] == '中文':
            # 设置中文模型
            set_language_config(config_file, 'ch')
        elif values['-LANGUAGE-'] == '日语':
            # 设置日语模型
            set_language_config(config_file, 'japan')
        elif values['-LANGUAGE-'] == '韩语':
            # 设置韩语模型
            set_language_config(config_file, 'korean')
        elif values['-LANGUAGE-'] == '法语':
            # 设置法语模型
            set_language_config(config_file, 'french')
        elif values['-LANGUAGE-'] == '德语':
            # 设置法语模型
            set_language_config(config_file, 'german')
        window.close()


def set_language_config(config_file, language_code):
    # 写入配置文件
    with open(config_file, mode='w') as f:
        f.write('[DEFAULT]\n')
        f.write(f'Language = {language_code}')


