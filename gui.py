# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao 
@Time    : 2021/4/1 6:07 下午
@FileName: gui.py
@desc: 字幕提取器图形化界面
"""
import warnings
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dependencies'))
import configparser
import PySimpleGUI as sg
import cv2
from threading import Thread


class SubtitleExtractorGUI:
    def __init__(self):
        sg.theme('LightBrown12')
        self.config_file = os.path.join(os.path.dirname(__file__), 'settings.ini')
        self.config = configparser.ConfigParser()
        self.interface_config = configparser.ConfigParser()
        if not os.path.exists(self.config_file):
            # 如果没有配置文件，默认弹出语言选择界面
            LanguageModeGUI().run()
        self.INTERFACE_KEY_NAME_MAP = {
            '简体中文': 'ch',
            '繁體中文': 'ch_tra',
            'English': 'en',
        }
        self.config.read(self.config_file, encoding='utf-8')
        self.interface_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'interface',
                                           f"{self.INTERFACE_KEY_NAME_MAP[self.config['DEFAULT']['Interface']]}.ini")
        self.interface_config.read(self.interface_file, encoding='utf-8')
        # 获取窗口分辨率
        self.screen_width, self.screen_height = sg.Window.get_screen_size()
        # 设置视频预览区域大小
        self.video_preview_height = self.screen_height // 2
        self.video_preview_width = self.screen_width // 2
        # 字幕提取器布局
        self.layout = None
        # 字幕提取其窗口
        self.window = None
        # 视频路径
        self.video_path = None
        # 视频cap
        self.video_cap = None
        # 视频的帧率
        self.fps = None
        # 视频的帧数
        self.frame_count = None
        # 视频的宽
        self.frame_width = None
        # 视频的高
        self.frame_height = None
        # 设置字幕区域高宽
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        # 字幕提取器
        self.se = None

    def run(self):
        # 创建布局
        self._create_layout()
        # 创建窗口
        self.window = sg.Window(title=self.interface_config['SubtitleExtractorGUI']['Title'], layout=self.layout)
        while True:
            # 循环读取事件
            event, values = self.window.read(timeout=10)
            # 处理【打开】事件
            self._file_event_handler(event, values)
            # 处理【滑动】事件
            self._slide_event_handler(event, values)
            # 处理【识别语言】事件
            self._language_mode_event_handler(event)
            # 处理【运行】事件
            self._run_event_handler(event, values)
            # 如果关闭软件，退出
            if event == sg.WIN_CLOSED:
                break
            # 更新进度条
            if self.se is not None:
                self.window['-PROG-'].update(self.se.progress)
                if self.se.isFinished:
                    # 1) 打开修改字幕滑块区域按钮
                    self.window['-Y-SLIDER-'].update(disabled=False)
                    self.window['-X-SLIDER-'].update(disabled=False)
                    self.window['-Y-SLIDER-H-'].update(disabled=False)
                    self.window['-X-SLIDER-W-'].update(disabled=False)
                    # 2) 打开【运行】、【打开】和【识别语言】按钮
                    self.window['-RUN-'].update(disabled=False)
                    self.window['-FILE-'].update(disabled=False)
                    self.window['-FILE_BTN-'].update(disabled=False)
                    self.window['-LANGUAGE-MODE-'].update(disabled=False)


    def _create_layout(self):
        """
        创建字幕提取器布局
        """
        self.layout = [
            # 显示视频预览
            [sg.Image(size=(self.video_preview_width, self.video_preview_height), background_color='black',
                      key='-DISPLAY-')],
            # 打开按钮 + 快进快退条
            [sg.Input(key='-FILE-', visible=False, enable_events=True),
             sg.FileBrowse(self.interface_config['SubtitleExtractorGUI']['Open'], file_types=((
                           self.interface_config['SubtitleExtractorGUI']['AllFile'], '*.*'), ('mp4', '*.mp4'),
                                                                                              ('flv', '*.flv'),
                                                                                              ('wmv', '*.wmv'),
                                                                                              ('avi', '*.avi')),
                           key='-FILE_BTN-'),
             sg.Slider(size=(80, 20), range=(1, 1), key='-SLIDER-', orientation='h',
                       enable_events=True,
                       disable_number_display=True),
             ],
            # 输出区域
            [sg.Output(size=(70, 10), font='Courier 10'),
             sg.Frame(title=self.interface_config['SubtitleExtractorGUI']['Vertical'], layout=[[
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 20),
                           # disable_number_display=True,
                           enable_events=True,
                           default_value=0, key='-Y-SLIDER-'),
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 20),
                           # disable_number_display=True,
                           enable_events=True,
                           default_value=0, key='-Y-SLIDER-H-'),
             ]], pad=((15, 5), (0, 0))),
             sg.Frame(title=self.interface_config['SubtitleExtractorGUI']['Horizontal'], layout=[[
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 20),
                           # disable_number_display=True,
                           enable_events=True,
                           default_value=0, key='-X-SLIDER-'),
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 20),
                           # disable_number_display=True,
                           enable_events=True,
                           default_value=0, key='-X-SLIDER-W-'),
             ]], pad=((15, 5), (0, 0)))
             ],

            # 运行按钮 + 进度条
            [sg.Button(button_text=self.interface_config['SubtitleExtractorGUI']['Run'], key='-RUN-', size=(20, 1)),
             sg.Button(button_text=self.interface_config['SubtitleExtractorGUI']['Setting'], key='-LANGUAGE-MODE-',
                       size=(20, 1)),
             sg.ProgressBar(100, orientation='h', size=(44, 20), key='-PROG-')
             ],
        ]

    def _file_event_handler(self, event, values):
        """
        当点击打开按钮时：
        1）打开视频文件，将画布显示视频帧
        2）获取视频信息，初始化进度条滑块范围
        """
        if event == '-FILE-':
            self.video_path = values['-FILE-']
            if self.video_path != '':
                self.video_cap = cv2.VideoCapture(self.video_path)
            if self.video_cap is None:
                return
            if self.video_cap.isOpened():
                ret, frame = self.video_cap.read()
                if ret:
                    print(f"{self.interface_config['SubtitleExtractorGUI']['OpenVideoSuccess']}：{self.video_path}")
                    # 获取视频的帧数
                    self.frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    # 获取视频的高度
                    self.frame_height = self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    # 获取视频的宽度
                    self.frame_width = self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    # 获取视频的帧率
                    self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                    # 调整视频帧大小，是播放器能够显示
                    resized_frame = cv2.resize(src=frame, dsize=(self.video_preview_width, self.video_preview_height))
                    # 显示视频帧
                    self.window['-DISPLAY-'].update(data=cv2.imencode('.png', resized_frame)[1].tobytes())
                    # 更新视频进度条滑块range
                    self.window['-SLIDER-'].update(range=(1, self.frame_count))
                    self.window['-SLIDER-'].update(1)
                    # 更新视频字幕位置滑块range
                    self.window['-Y-SLIDER-'].update(range=(0, self.frame_height), disabled=False)
                    self.window['-Y-SLIDER-H-'].update(range=(0, self.frame_height // 2), disabled=False)
                    self.window['-Y-SLIDER-'].update(self.frame_height * .85)
                    self.window['-Y-SLIDER-H-'].update(self.frame_height * .146)
                    self.window['-X-SLIDER-'].update(range=(0, self.frame_width), disabled=False)
                    self.window['-X-SLIDER-W-'].update(range=(0, self.frame_width), disabled=False)
                    self.window['-X-SLIDER-'].update(self.frame_width * .15)
                    self.window['-X-SLIDER-W-'].update(self.frame_width * .7)

    @staticmethod
    def _language_mode_event_handler(event):
        if event != '-LANGUAGE-MODE-':
            return
        if 'OK' == LanguageModeGUI().run():
            # 重新加载config
            from backend.main import SubtitleExtractor

    def _run_event_handler(self, event, values):
        """
        当点击运行按钮时：
        1) 禁止修改字幕滑块区域
        2) 禁止再次点击【运行】和【打开】按钮
        3) 设定字幕区域位置
        """
        if event == '-RUN-':
            if self.video_cap is None:
                print(self.interface_config['SubtitleExtractorGUI']['OpenVideoFirst'])
            else:
                # 1) 禁止修改字幕滑块区域
                self.window['-Y-SLIDER-'].update(disabled=True)
                self.window['-X-SLIDER-'].update(disabled=True)
                self.window['-Y-SLIDER-H-'].update(disabled=True)
                self.window['-X-SLIDER-W-'].update(disabled=True)
                # 2) 禁止再次点击【运行】、【打开】和【识别语言】按钮
                self.window['-RUN-'].update(disabled=True)
                self.window['-FILE-'].update(disabled=True)
                self.window['-FILE_BTN-'].update(disabled=True)
                self.window['-LANGUAGE-MODE-'].update(disabled=True)
                # 3) 设定字幕区域位置
                self.xmin = int(values['-X-SLIDER-'])
                self.xmax = int(values['-X-SLIDER-'] + values['-X-SLIDER-W-'])
                self.ymin = int(values['-Y-SLIDER-'])
                self.ymax = int(values['-Y-SLIDER-'] + values['-Y-SLIDER-H-'])
                print(f"{self.interface_config['SubtitleExtractorGUI']['SubtitleArea']}：({self.ymin},{self.ymax},{self.xmin},{self.xmax})")
                subtitle_area = (self.ymin, self.ymax, self.xmin, self.xmax)
                from backend.main import SubtitleExtractor
                self.se = SubtitleExtractor(self.video_path, subtitle_area)
                Thread(target=self.se.run, daemon=True).start()

    def _slide_event_handler(self, event, values):
        """
        当滑动视频进度条/滑动字幕选择区域滑块时：
        1) 判断视频是否存在，如果存在则显示对应的视频帧
        2) 绘制rectangle
        """
        if event == '-SLIDER-' or event == '-Y-SLIDER-' or event == '-Y-SLIDER-H-' or event == '-X-SLIDER-' or event \
                == '-X-SLIDER-W-':
            if self.video_cap is not None and self.video_cap.isOpened():
                frame_no = int(values['-SLIDER-'])
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = self.video_cap.read()
                if ret:
                    # 画字幕框
                    y = int(values['-Y-SLIDER-'])
                    h = int(values['-Y-SLIDER-H-'])
                    x = int(values['-X-SLIDER-'])
                    w = int(values['-X-SLIDER-W-'])
                    draw = cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h),
                                         color=(0, 255, 0), thickness=3)
                    # 调整视频帧大小，使播放器能够显示
                    resized_frame = cv2.resize(src=draw, dsize=(self.video_preview_width, self.video_preview_height))
                    # 显示视频帧
                    self.window['-DISPLAY-'].update(data=cv2.imencode('.png', resized_frame)[1].tobytes())


class LanguageModeGUI:
    def __init__(self):
        self.config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.ini')
        # 设置界面
        self.INTERFACE_DEF = '简体中文'
        if not os.path.exists(self.config_file):
            self.interface_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'interface',
                                               "ch.ini")
        self.interface_config = configparser.ConfigParser()
        # 设置语言
        self.INTERFACE_KEY_NAME_MAP = {
            '简体中文': 'ch',
            '繁體中文': 'ch_tra',
            'English': 'en',
        }
        # 设置语言
        self.LANGUAGE_DEF = 'ch'
        self.LANGUAGE_NAME_KEY_MAP = None
        self.LANGUAGE_KEY_NAME_MAP = None
        self.MODE_DEF = 'fast'
        self.MODE_NAME_KEY_MAP = None
        self.MODE_KEY_NAME_MAP = None
        # 语言选择布局
        self.layout = None
        # 语言选择窗口
        self.window = None

    def run(self):
        # 创建布局
        title = self._create_layout()
        # 创建窗口
        self.window = sg.Window(title=title, layout=self.layout)
        while True:
            # 循环读取事件
            event, values = self.window.read(timeout=10)
            # 处理【OK】事件
            self._ok_event_handler(event, values)
            # 处理【切换界面语言】事件
            self._interface_event_handler(event, values)
            # 如果关闭软件，退出
            if event == sg.WIN_CLOSED:
                if os.path.exists(self.config_file):
                    break
                else:
                    exit(0)
            if event == 'Cancel':
                if os.path.exists(self.config_file):
                    self.window.close()
                    break
                else:
                    exit(0)

    def _load_interface_text(self):
        self.interface_config.read(self.interface_file, encoding='utf-8')
        # 设置界面
        self.INTERFACE_DEF = self.interface_config["LanguageModeGUI"]["InterfaceDefault"]

        self.LANGUAGE_DEF = self.interface_config["LanguageModeGUI"]["LanguageSimplifiedChinese"]
        self.LANGUAGE_NAME_KEY_MAP = {
            self.interface_config["LanguageModeGUI"]["LanguageSimplifiedChinese"]: 'ch',
            self.interface_config["LanguageModeGUI"]["LanguageTraditionalChinese"]: 'ch_tra',
            self.interface_config["LanguageModeGUI"]["LanguageEnglish"]: 'en',
            self.interface_config["LanguageModeGUI"]["LanguageJapanese"]: 'japan',
            self.interface_config["LanguageModeGUI"]["LanguageKorean"]: 'korean',
            self.interface_config["LanguageModeGUI"]["LanguageFrench"]: 'french',
            self.interface_config["LanguageModeGUI"]["LanguageGerman"]: 'german',
            self.interface_config["LanguageModeGUI"]["LanguageRussian"]: 'ru',
            self.interface_config["LanguageModeGUI"]["LanguageSpanish"]: 'es',
            self.interface_config["LanguageModeGUI"]["LanguagePortuguese"]: 'pt',
            self.interface_config["LanguageModeGUI"]["LanguageItalian"]: 'it',
        }

        self.LANGUAGE_KEY_NAME_MAP = {v: k for k, v in self.LANGUAGE_NAME_KEY_MAP.items()}
        self.MODE_DEF = self.interface_config["LanguageModeGUI"]['ModeFast']
        self.MODE_NAME_KEY_MAP = {
            self.interface_config["LanguageModeGUI"]['ModeFast']: 'fast',
            self.interface_config["LanguageModeGUI"]['ModeAccurate']: 'accurate',
        }
        self.MODE_KEY_NAME_MAP = {v: k for k, v in self.MODE_NAME_KEY_MAP.items()}

    def _create_layout(self):
        interface_def, language_def, mode_def = self.parse_config(self.config_file)
        # 加载界面文本
        self._load_interface_text()
        choose_language_text = self.interface_config["LanguageModeGUI"]["InterfaceLanguage"]
        choose_sub_lang_text = self.interface_config["LanguageModeGUI"]["SubtitleLanguage"]
        choose_mode_text = self.interface_config["LanguageModeGUI"]["Mode"]
        self.layout = [
            # 显示选择界面语言
            [sg.Text(choose_language_text),
             sg.DropDown(values=list(self.INTERFACE_KEY_NAME_MAP.keys()), size=(30, 20),
                         pad=(0, 20),
                         key='-INTERFACE-', readonly=True,
                         default_value=interface_def),
             sg.OK(key='-INTERFACE-OK-')],
            # 显示选择字幕语言
            [sg.Text(choose_sub_lang_text),
             sg.DropDown(values=list(self.LANGUAGE_NAME_KEY_MAP.keys()), size=(30, 20),
                         pad=(0, 20),
                         key='-LANGUAGE-', readonly=True, default_value=language_def)],
            # 显示识别模式
            [sg.Text(choose_mode_text),
             sg.DropDown(values=list(self.MODE_NAME_KEY_MAP.keys()), size=(30, 20), pad=(0, 20),
                         key='-MODE-', readonly=True, default_value=mode_def)],
            # 显示确认关闭按钮
            [sg.OK(), sg.Cancel()]
        ]
        return self.interface_config["LanguageModeGUI"]["Title"]

    def _ok_event_handler(self, event, values):
        if event == 'OK':
            # 设置模型语言配置
            interface = None
            language = None
            mode = None
            # 设置界面语言
            interface_str = values['-INTERFACE-']
            if interface_str in self.INTERFACE_KEY_NAME_MAP:
                interface = interface_str
            language_str = values['-LANGUAGE-']
            # 设置字幕语言
            print(self.interface_config["LanguageModeGUI"]["SubtitleLanguage"], language_str)
            if language_str in self.LANGUAGE_NAME_KEY_MAP:
                language = self.LANGUAGE_NAME_KEY_MAP[language_str]
            # 设置模型语言配置
            mode_str = values['-MODE-']
            print(self.interface_config["LanguageModeGUI"]["Mode"], mode_str)
            if mode_str in self.MODE_NAME_KEY_MAP:
                mode = self.MODE_NAME_KEY_MAP[mode_str]
            self.set_config(self.config_file, interface, language, mode)
            self.window.close()

    def _interface_event_handler(self, event, values):
        if event == '-INTERFACE-OK-':
            self.interface_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'interface',
                                               f"{self.INTERFACE_KEY_NAME_MAP[values['-INTERFACE-']]}.ini")
            self.interface_config.read(self.interface_file, encoding='utf-8')
            config = configparser.ConfigParser()
            if os.path.exists(self.config_file):
                config.read(self.config_file, encoding='utf-8')
                self.set_config(self.config_file, values['-INTERFACE-'], config['DEFAULT']['Language'],
                                config['DEFAULT']['Mode'])
            self.window.close()
            title = self._create_layout()
            self.window = sg.Window(title=title, layout=self.layout)

    @staticmethod
    def set_config(config_file, interface, language_code, mode):
        # 写入配置文件
        with open(config_file, mode='w', encoding='utf-8') as f:
            f.write('[DEFAULT]\n')
            f.write(f'Interface = {interface}\n')
            f.write(f'Language = {language_code}\n')
            f.write(f'Mode = {mode}\n')

    def parse_config(self, config_file):
        if not os.path.exists(config_file):
            self.interface_config.read(self.interface_file, encoding='utf-8')
            interface_def = self.interface_config['LanguageModeGUI']['InterfaceDefault']
            language_def = self.interface_config['LanguageModeGUI']['InterfaceDefault']
            mode_def = self.interface_config['LanguageModeGUI']['ModeFast']
            return interface_def, language_def, mode_def
        config = configparser.ConfigParser()
        config.read(config_file, encoding='utf-8')
        interface = config['DEFAULT']['Interface']
        language = config['DEFAULT']['Language']
        mode = config['DEFAULT']['Mode']
        self.interface_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'interface',
                                           f"{self.INTERFACE_KEY_NAME_MAP[interface]}.ini")
        self._load_interface_text()
        interface_def = interface if interface in self.INTERFACE_KEY_NAME_MAP else \
            self.INTERFACE_DEF
        language_def = self.LANGUAGE_KEY_NAME_MAP[language] if language in self.LANGUAGE_KEY_NAME_MAP else \
            self.LANGUAGE_DEF
        mode_def = self.MODE_KEY_NAME_MAP[mode] if mode in self.MODE_KEY_NAME_MAP else self.MODE_DEF
        return interface_def, language_def, mode_def


if __name__ == '__main__':
    # 运行图形化界面
    subtitleExtractorGUI = SubtitleExtractorGUI()
    subtitleExtractorGUI.run()
