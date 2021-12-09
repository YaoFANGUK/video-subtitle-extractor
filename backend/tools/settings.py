# -*- coding: UTF-8 -*-  
"""
@author: Fang Yao
@file  : settings.py
@time  : 2021/09/10 09:20
@desc  : 
"""
import configparser
import os
import PySimpleGUI as sg

LANGUAGE_DEF = '中文/英文'
LANGUAGE_NAME_KEY_MAP = {
    '中文/英文':    'ch',
    '繁体中文':     'ch_tra',
    '日语':         'japan',
    '韩语':         'korean',
    '法语':         'french',
    '德语':         'german',
}
LANGUAGE_KEY_NAME_MAP = {v: k for k, v in LANGUAGE_NAME_KEY_MAP.items()}
MODE_DEF = '快速'
MODE_NAME_KEY_MAP = {
    '快速':          'fast',
    '精准':          'accurate',
}
MODE_KEY_NAME_MAP = {v: k for k, v in MODE_NAME_KEY_MAP.items()}


def set_language_mode(config_file):
    language_def, mode_def = parse_config(config_file)
    window = sg.Window(title='字幕提取器',
                       layout=[
                           [sg.Text('选择视频字幕的语言:'), sg.DropDown(values=list(LANGUAGE_NAME_KEY_MAP.keys()), size=(30, 20), pad=(0, 20),
                                                               key='-LANGUAGE-', readonly=True, default_value=language_def)],
                           [sg.Text('选择识别模式:'), sg.DropDown(values=list(MODE_NAME_KEY_MAP.keys()), size=(30, 20), pad=(0, 20),
                                                            key='-MODE-', readonly=True, default_value=mode_def)],
                           [sg.OK(), sg.Cancel()]
                       ])
    event, values = window.read()
    if event == 'OK':
        # 设置模型语言配置
        language = None
        mode = None
        language_str = values['-LANGUAGE-']
        print('选择语言:', language_str)
        if language_str in LANGUAGE_NAME_KEY_MAP:
            language = LANGUAGE_NAME_KEY_MAP[language_str]
        # 设置模型语言配置
        mode_str = values['-MODE-']
        print('选择模式:', mode_str)
        if mode_str in MODE_NAME_KEY_MAP:
            mode = MODE_NAME_KEY_MAP[mode_str]
        set_config(config_file, language, mode)
    window.close()
    return event


def set_config(config_file, language_code, mode):
    # 写入配置文件
    with open(config_file, mode='w') as f:
        f.write('[DEFAULT]\n')
        f.write(f'Language = {language_code}\n')
        f.write(f'Mode = {mode}\n')


def parse_config(config_file):
    if not os.path.exists(config_file):
        return LANGUAGE_DEF, MODE_DEF
    config = configparser.ConfigParser()
    config.read(config_file)
    language = config['DEFAULT']['Language']
    mode = config['DEFAULT']['Mode']
    language_def = LANGUAGE_KEY_NAME_MAP[language] if language in LANGUAGE_KEY_NAME_MAP else LANGUAGE_DEF
    mode_def = MODE_KEY_NAME_MAP[mode] if mode in MODE_KEY_NAME_MAP else MODE_DEF
    return language_def, mode_def
