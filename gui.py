# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao
@Time    : 2021/4/1 6:07 下午
@FileName: gui.py
@desc: 字幕提取器图形化界面
"""
import backend.main
import os
import configparser
import PySimpleGUI as sg
import cv2
from threading import Thread
import multiprocessing


class SubtitleExtractorGUI:
    def _load_config(self):
        self.config_file = os.path.join(os.path.dirname(__file__), 'settings.ini')
        self.subtitle_config_file = os.path.join(os.path.dirname(__file__), 'subtitle.ini')
        self.config = configparser.ConfigParser()
        self.interface_config = configparser.ConfigParser()
        if not os.path.exists(self.config_file):
            # 如果没有配置文件，默认弹出语言选择界面
            LanguageModeGUI(self).run()
        self.INTERFACE_KEY_NAME_MAP = {
            '简体中文': 'ch',
            '繁體中文': 'chinese_cht',
            'English': 'en',
            '한국어': 'ko',
        }
        self.config.read(self.config_file, encoding='utf-8')
        self.interface_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'interface',
                                           f"{self.INTERFACE_KEY_NAME_MAP[self.config['DEFAULT']['Interface']]}.ini")
        self.interface_config.read(self.interface_file, encoding='utf-8')

    def __init__(self):
        # 初次运行检查运行环境是否正常
        from paddle import fluid
        fluid.install_check.run_check()
        self.font = 'Arial 10'
        self.theme = 'LightBrown12'
        sg.theme(self.theme)
        self.icon = os.path.join(os.path.dirname(__file__), 'design', 'vse.ico')
        self._load_config()
        self.screen_width, self.screen_height = sg.Window.get_screen_size()
        print(self.screen_width, self.screen_height)
        # 设置视频预览区域大小
        self.video_preview_width = 960
        self.video_preview_height = self.video_preview_width * 9 // 16
        # 默认组件大小
        self.horizontal_slider_size = (120, 20)
        self.output_size = (100, 10)
        self.progressbar_size = (60, 20)
        # 分辨率低于1080
        if self.screen_width // 2 < 960:
            self.video_preview_width = 640
            self.video_preview_height = self.video_preview_width * 9 // 16
            self.horizontal_slider_size = (60, 20)
            self.output_size = (58, 10)
            self.progressbar_size = (28, 20)
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
        self.window = sg.Window(title=self.interface_config['SubtitleExtractorGUI']['Title'], layout=self.layout,
                                icon=self.icon)
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
                self.window['-PROG-'].update(self.se.progress_total)
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
                    self.se = None
                if len(self.video_paths) >= 1:
                    # 1) 关闭修改字幕滑块区域按钮
                    self.window['-Y-SLIDER-'].update(disabled=True)
                    self.window['-X-SLIDER-'].update(disabled=True)
                    self.window['-Y-SLIDER-H-'].update(disabled=True)
                    self.window['-X-SLIDER-W-'].update(disabled=True)
                    # 2) 关闭【运行】、【打开】和【识别语言】按钮
                    self.window['-RUN-'].update(disabled=True)
                    self.window['-FILE-'].update(disabled=True)
                    self.window['-FILE_BTN-'].update(disabled=True)
                    self.window['-LANGUAGE-MODE-'].update(disabled=True)

    def update_interface_text(self):
        self._load_config()
        self.window.set_title(self.interface_config['SubtitleExtractorGUI']['Title'])
        self.window['-FILE_BTN-'].Update(self.interface_config['SubtitleExtractorGUI']['Open'])
        self.window['-FRAME1-'].Update(self.interface_config['SubtitleExtractorGUI']['Vertical'])
        self.window['-FRAME2-'].Update(self.interface_config['SubtitleExtractorGUI']['Horizontal'])
        self.window['-RUN-'].Update(self.interface_config['SubtitleExtractorGUI']['Run'])
        self.window['-LANGUAGE-MODE-'].Update(self.interface_config['SubtitleExtractorGUI']['Setting'])

    def _create_layout(self):
        """
        创建字幕提取器布局
        """
        garbage = os.path.join(os.path.dirname(__file__), 'output')
        if os.path.exists(garbage):
            import shutil
            shutil.rmtree(garbage, True)
        self.layout = [
            # 显示视频预览
            [sg.Image(size=(self.video_preview_width, self.video_preview_height), background_color='black',
                      key='-DISPLAY-')],
            # 打开按钮 + 快进快退条
            [sg.Input(key='-FILE-', visible=False, enable_events=True),
             sg.FilesBrowse(button_text=self.interface_config['SubtitleExtractorGUI']['Open'], file_types=((
                            self.interface_config['SubtitleExtractorGUI']['AllFile'], '*.*'), ('mp4', '*.mp4'),
                                                                                              ('flv', '*.flv'),
                                                                                              ('wmv', '*.wmv'),
                                                                                              ('avi', '*.avi')),
                            key='-FILE_BTN-', size=(10, 1), font=self.font),
             sg.Slider(size=self.horizontal_slider_size, range=(1, 1), key='-SLIDER-', orientation='h',
                       enable_events=True, font=self.font,
                       disable_number_display=True),
             ],
            # 输出区域
            [sg.Output(size=self.output_size, font=self.font),
             sg.Frame(title=self.interface_config['SubtitleExtractorGUI']['Vertical'], font=self.font, key='-FRAME1-',
             layout=[[
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 20),
                           disable_number_display=True,
                           enable_events=True, font=self.font,
                           pad=((10, 10), (20, 20)),
                           default_value=0, key='-Y-SLIDER-'),
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 20),
                           disable_number_display=True,
                           enable_events=True, font=self.font,
                           pad=((10, 10), (20, 20)),
                           default_value=0, key='-Y-SLIDER-H-'),
             ]], pad=((15, 5), (0, 0))),
             sg.Frame(title=self.interface_config['SubtitleExtractorGUI']['Horizontal'], font=self.font, key='-FRAME2-',
             layout=[[
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 20),
                           disable_number_display=True,
                           pad=((10, 10), (20, 20)),
                           enable_events=True, font=self.font,
                           default_value=0, key='-X-SLIDER-'),
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 20),
                           disable_number_display=True,
                           pad=((10, 10), (20, 20)),
                           enable_events=True, font=self.font,
                           default_value=0, key='-X-SLIDER-W-'),
             ]], pad=((15, 5), (0, 0)))
             ],

            # 运行按钮 + 进度条
            [sg.Button(button_text=self.interface_config['SubtitleExtractorGUI']['Run'], key='-RUN-',
                       font=self.font, size=(20, 1)),
             sg.Button(button_text=self.interface_config['SubtitleExtractorGUI']['Setting'], key='-LANGUAGE-MODE-',
                       font=self.font, size=(20, 1)),
             sg.ProgressBar(100, orientation='h', size=self.progressbar_size, key='-PROG-', auto_size_text=True)
             ],
        ]

    def _file_event_handler(self, event, values):
        """
        当点击打开按钮时：
        1）打开视频文件，将画布显示视频帧
        2）获取视频信息，初始化进度条滑块范围
        """
        if event == '-FILE-':
            self.video_paths = values['-FILE-'].split(';')
            self.video_path = self.video_paths[0]
            if self.video_path != '':
                self.video_cap = cv2.VideoCapture(self.video_path)
            if self.video_cap is None:
                return
            if self.video_cap.isOpened():
                ret, frame = self.video_cap.read()
                if ret:
                    for video in self.video_paths:
                        print(f"{self.interface_config['SubtitleExtractorGUI']['OpenVideoSuccess']}：{video}")
                    # 获取视频的帧数
                    self.frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    # 获取视频的高度
                    self.frame_height = self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    # 获取视频的宽度
                    self.frame_width = self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    # 获取视频的帧率
                    self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                    # 调整视频帧大小，使播放器能够显示
                    resized_frame = self._img_resize(frame)
                    # resized_frame = cv2.resize(src=frame, dsize=(self.video_preview_width, self.video_preview_height))
                    # 显示视频帧
                    self.window['-DISPLAY-'].update(data=cv2.imencode('.png', resized_frame)[1].tobytes())
                    # 更新视频进度条滑块range
                    self.window['-SLIDER-'].update(range=(1, self.frame_count))
                    self.window['-SLIDER-'].update(1)
                    # 预设字幕区域位置
                    y_p, h_p, x_p, w_p = self.parse_subtitle_config()
                    y = self.frame_height * y_p
                    h = self.frame_height * h_p
                    x = self.frame_width * x_p
                    w = self.frame_width * w_p
                    # 更新视频字幕位置滑块range
                    # 更新Y-SLIDER范围
                    self.window['-Y-SLIDER-'].update(range=(0, self.frame_height), disabled=False)
                    # 更新Y-SLIDER默认值
                    self.window['-Y-SLIDER-'].update(y)
                    # 更新X-SLIDER范围
                    self.window['-X-SLIDER-'].update(range=(0, self.frame_width), disabled=False)
                    # 更新X-SLIDER默认值
                    self.window['-X-SLIDER-'].update(x)
                    # 更新Y-SLIDER-H范围
                    self.window['-Y-SLIDER-H-'].update(range=(0, self.frame_height - y))
                    # 更新Y-SLIDER-H默认值
                    self.window['-Y-SLIDER-H-'].update(h)
                    # 更新X-SLIDER-W范围
                    self.window['-X-SLIDER-W-'].update(range=(0, self.frame_width - x))
                    # 更新X-SLIDER-W默认值
                    self.window['-X-SLIDER-W-'].update(w)
                    self._update_preview(frame, (y, h, x, w))

    def _language_mode_event_handler(self, event):
        if event != '-LANGUAGE-MODE-':
            return
        if 'OK' == LanguageModeGUI(self).run():
            # 重新加载config
            pass

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
                if self.ymax > self.frame_height:
                    self.ymax = self.frame_height
                if self.xmax > self.frame_width:
                    self.xmax = self.frame_width
                print(f"{self.interface_config['SubtitleExtractorGUI']['SubtitleArea']}：({self.ymin},{self.ymax},{self.xmin},{self.xmax})")
                subtitle_area = (self.ymin, self.ymax, self.xmin, self.xmax)
                y_p = self.ymin / self.frame_height
                h_p = (self.ymax - self.ymin) / self.frame_height
                x_p = self.xmin / self.frame_width
                w_p = (self.xmax - self.xmin) / self.frame_width
                self.set_subtitle_config(y_p, h_p, x_p, w_p)

                def task():
                    while self.video_paths:
                        video_path = self.video_paths.pop()
                        self.se = backend.main.SubtitleExtractor(video_path, subtitle_area)
                        self.se.run()
                Thread(target=task, daemon=True).start()
                self.video_cap.release()
                self.video_cap = None

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
                    self.window['-Y-SLIDER-H-'].update(range=(0, self.frame_height-values['-Y-SLIDER-']))
                    self.window['-X-SLIDER-W-'].update(range=(0, self.frame_width-values['-X-SLIDER-']))
                    # 画字幕框
                    y = int(values['-Y-SLIDER-'])
                    h = int(values['-Y-SLIDER-H-'])
                    x = int(values['-X-SLIDER-'])
                    w = int(values['-X-SLIDER-W-'])
                    self._update_preview(frame, (y, h, x, w))

    def _update_preview(self, frame, y_h_x_w):
        y, h, x, w = y_h_x_w
        # 画字幕框
        draw = cv2.rectangle(img=frame, pt1=(int(x), int(y)), pt2=(int(x) + int(w), int(y) + int(h)),
                             color=(0, 255, 0), thickness=3)
        # 调整视频帧大小，使播放器能够显示
        resized_frame = self._img_resize(draw)
        # 显示视频帧
        self.window['-DISPLAY-'].update(data=cv2.imencode('.png', resized_frame)[1].tobytes())


    def _img_resize(self, image):
        top, bottom, left, right = (0, 0, 0, 0)
        height, width = image.shape[0], image.shape[1]
        # 对长短不想等的图片，找到最长的一边
        longest_edge = height
        # 计算短边需要增加多少像素宽度使其与长边等长
        if width < longest_edge:
            dw = longest_edge - width
            left = dw // 2
            right = dw - left
        else:
            pass
        # 给图像增加边界
        constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return cv2.resize(constant, (self.video_preview_width, self.video_preview_height))

    def set_subtitle_config(self, y, h, x, w):
        # 写入配置文件
        with open(self.subtitle_config_file, mode='w', encoding='utf-8') as f:
            f.write('[AREA]\n')
            f.write(f'Y = {y}\n')
            f.write(f'H = {h}\n')
            f.write(f'X = {x}\n')
            f.write(f'W = {w}\n')

    def parse_subtitle_config(self):
        y_p, h_p, x_p, w_p = .78, .21, .05, .9
        # 如果配置文件不存在，则写入配置文件
        if not os.path.exists(self.subtitle_config_file):
            self.set_subtitle_config(y_p, h_p, x_p, w_p)
            return y_p, h_p, x_p, w_p
        else:
            try:
                config = configparser.ConfigParser()
                config.read(self.subtitle_config_file, encoding='utf-8')
                conf_y_p, conf_h_p, conf_x_p, conf_w_p = float(config['AREA']['Y']), float(config['AREA']['H']), float(config['AREA']['X']), float(config['AREA']['W'])
                return conf_y_p, conf_h_p, conf_x_p, conf_w_p
            except Exception:
                self.set_subtitle_config(y_p, h_p, x_p, w_p)
                return y_p, h_p, x_p, w_p


class LanguageModeGUI:
    def __init__(self, subtitle_extractor_gui):
        self.subtitle_extractor_gui = subtitle_extractor_gui
        self.icon = os.path.join(os.path.dirname(__file__), 'design', 'vse.ico')
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
            '繁體中文': 'chinese_cht',
            'English': 'en',
            '한국어': 'ko',
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
        self.window = sg.Window(title=title, layout=self.layout, icon=self.icon)
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
        config_language_mode_gui = self.interface_config["LanguageModeGUI"]
        # 设置界面
        self.INTERFACE_DEF = config_language_mode_gui["InterfaceDefault"]

        self.LANGUAGE_DEF = config_language_mode_gui["LanguageCH"]
        self.LANGUAGE_NAME_KEY_MAP = {}
        for lang in backend.main.config.MULTI_LANG:
            self.LANGUAGE_NAME_KEY_MAP[config_language_mode_gui[f"Language{lang.upper()}"]] = lang
        self.LANGUAGE_NAME_KEY_MAP = dict(sorted(self.LANGUAGE_NAME_KEY_MAP.items(), key=lambda item: item[1]))
        self.LANGUAGE_KEY_NAME_MAP = {v: k for k, v in self.LANGUAGE_NAME_KEY_MAP.items()}
        self.MODE_DEF = config_language_mode_gui['ModeFast']
        self.MODE_NAME_KEY_MAP = {
            config_language_mode_gui['ModeAuto']: 'auto',
            config_language_mode_gui['ModeFast']: 'fast',
            config_language_mode_gui['ModeAccurate']: 'accurate',
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
            if self.subtitle_extractor_gui is not None:
                self.subtitle_extractor_gui.update_interface_text()
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
            self.window = sg.Window(title=title, layout=self.layout, icon=self.icon)

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
    try:
        multiprocessing.set_start_method("spawn")
        # 运行图形化界面
        subtitleExtractorGUI = SubtitleExtractorGUI()
        subtitleExtractorGUI.run()
    except Exception as e:
        print(f'[{type(e)}] {e}')
        import traceback
        traceback.print_exc()
        msg = traceback.format_exc()
        err_log_path = os.path.join(os.path.expanduser('~'), 'VSE-Error-Message.log')
        with open(err_log_path, 'w', encoding='utf-8') as f:
            f.writelines(msg)
        import platform
        if platform.system() == 'Windows':
            os.system('pause')
        else:
            input()
