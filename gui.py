# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao 
@Time    : 2021/4/1 6:07 下午
@FileName: gui.py
@desc:
"""
import sys
import PySimpleGUI as sg
import os
import cv2
from PIL import Image
import io


class SubtitleExtractorGUI:
    def __init__(self):
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
        # 进度条最大值
        self.BAR_MAX = 1000

    def run(self):
        # 创建布局
        self._create_layout()
        # 创建窗口
        self.window = sg.Window('视频硬字幕提取器', self.layout)
        while True:
            # 循环读取事件
            event, values = self.window.read(timeout=10)
            # 处理【打开】事件
            self._file_event_handler(event, values)
            # 处理【滑动】事件
            self._slide_event_handler(event, values)
            # 处理【运行】事件
            self._run_event_handler(event, values)
            # 如果关闭软件，退出
            if event == sg.WIN_CLOSED:
                break

    def _create_layout(self):
        """
        创建字幕提取器布局
        """
        sg.theme('LightBrown12')
        self.layout = [
            # 菜单图标按钮
            [sg.Input(key='-FILE-', visible=False, enable_events=True),
             sg.FileBrowse('打开', file_types=(('mp4文件', '*.mp4'), ('flv文件', '*.flv'),
                                             ('wmv文件', '*.wmv'), ('avi文件', '*.avi')),
                           key='-FILE_BTN-'),
             sg.Button('执行'), sg.Button('设置'), sg.Button('帮助')],
            # 显示视频预览
            [sg.Image(size=(854, 480), background_color='black', key='-DISPLAY-')],
            # 快进快退条
            [sg.Slider(size=(120, 20), range=(1, 1), key='-SLIDER-', orientation='h', enable_events=True,
                       disable_number_display=True)
             ],
            # 输出区域
            [sg.Output(size=(100, 10), font='Courier 10'),
             sg.Frame(title='垂直方向', layout=[[
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 20),
                           # disable_number_display=True,
                           enable_events=True,
                           default_value=0, key='-Y-SLIDER-'),
                 sg.Slider(range=(0, 0), orientation='v', size=(10, 20),
                           # disable_number_display=True,
                           enable_events=True,
                           default_value=0, key='-Y-SLIDER-H-'),
             ]], pad=((15, 5), (0, 0))),
             sg.Frame(title='水平方向', layout=[[
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
            [sg.Button(button_text='运行', key='-RUN-'),
             sg.ProgressBar(max_value=self.BAR_MAX, orientation='h',
                            size=(90, 20), key='-PROG-', style='clam'
                            )],
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
            if self.video_cap.isOpened():
                ret, frame = self.video_cap.read()
                if ret:
                    print(f'成功打开视频：{self.video_path}')
                    # 获取视频的帧数
                    self.frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    # 获取视频的高度
                    self.frame_height = self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    # 获取视频的宽度
                    self.frame_width = self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    # 调整视频帧大小，是播放器能够显示
                    resized_frame = cv2.resize(src=frame, dsize=(854, 480))
                    # 显示视频帧
                    self.window['-DISPLAY-'].update(data=cv2.imencode('.png', resized_frame)[1].tobytes())
                    # 更新视频进度条滑块range
                    self.window['-SLIDER-'].update(range=(1, self.frame_count))
                    self.window['-SLIDER-'].update(1)
                    # 更新视频字幕位置滑块range
                    self.window['-Y-SLIDER-'].update(range=(0, self.frame_height), disabled=False)
                    self.window['-Y-SLIDER-H-'].update(range=(0, self.frame_height // 2), disabled=False)
                    self.window['-Y-SLIDER-'].update(self.frame_height - 50)
                    self.window['-Y-SLIDER-H-'].update(10)
                    self.window['-X-SLIDER-'].update(range=(0, self.frame_width), disabled=False)
                    self.window['-X-SLIDER-W-'].update(range=(0, self.frame_width), disabled=False)
                    self.window['-X-SLIDER-'].update(0)
                    self.window['-X-SLIDER-W-'].update(self.frame_width)

    def _run_event_handler(self, event, values):
        """
        当点击运行按钮时：
        1) 禁止修改字幕滑块区域
        2) 禁止再次点击【运行】和【打开】按钮
        3) 设定字幕区域位置
        """
        if event == '-RUN-':
            if self.video_cap is None:
                print('请先打开视频')
            else:
                # 1) 禁止修改字幕滑块区域
                self.window['-Y-SLIDER-'].update(disabled=True)
                self.window['-X-SLIDER-'].update(disabled=True)
                self.window['-Y-SLIDER-H-'].update(disabled=True)
                self.window['-X-SLIDER-W-'].update(disabled=True)
                # 2) 禁止再次点击【运行】和【打开】按钮
                self.window['-RUN-'].update(disabled=True)
                self.window['-FILE-'].update(disabled=True)
                self.window['-FILE_BTN-'].update(disabled=True)
                # 3) 设定字幕区域位置
                self.xmin = values['-X-SLIDER-']
                self.xmax = values['-X-SLIDER-'] + values['-X-SLIDER-W-']
                self.ymin = values['-Y-SLIDER-']
                self.ymax = values['-Y-SLIDER-'] + values['-Y-SLIDER-H-']
                print(f'字幕区域：({self.ymin},{self.ymax},{self.xmin},{self.xmax})')

    def _slide_event_handler(self, event, values):
        """
        当滑动视频进度条/滑动字幕选择区域滑块时：
        1) 判断视频是否存在，如果存在则显示对应的视频帧
        2) 绘制rectangle
        """
        if event == '-SLIDER-' or event == '-Y-SLIDER-' or event == '-Y-SLIDER-H-' or event == '-X-SLIDER-' or event == '-X-SLIDER-W-':
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
                    # 调整视频帧大小，是播放器能够显示
                    resized_frame = cv2.resize(src=draw, dsize=(854, 480))
                    # 显示视频帧
                    self.window['-DISPLAY-'].update(data=cv2.imencode('.png', resized_frame)[1].tobytes())


if __name__ == '__main__':
    # 创建GUI对象
    subtitleExtractorGUI = SubtitleExtractorGUI()
    # 运行字幕提取器
    subtitleExtractorGUI.run()
