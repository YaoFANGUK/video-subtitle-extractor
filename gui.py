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


class GUI:
    def __init__(self):
        self.play_pause_btn_base64 = b'iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAQAAABKfvVzAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAAmJLR0QAAKqNIzIAAAAJcEhZcwAA6mAAAOpgAYTJ3nYAAAAHdElNRQflBAcJJRr4Sqz6AAABT0lEQVQ4y43QP0hVcRTA8c/v3vdSe4Lh4uBgQ04NLenu2lq5KYghLbb1B2qJSN6go1hEODSYgmtLw6NGlaC9wYigCNyiP+/ddxtM6N7u+z3PdM7vnO/h+zspgrqu3jFgSBCclcpSwZI7vjnsCSxrSpzz1EVvaGjJfXHLUA9gQ27VvNxbw4kgYMyadROVQBdd+XGW/H088N2CHTP6xAnwxKIPpm1F1ApA265rXkXVCkDAe3OafsXVkkJ15IEbcbWkVGdextWSiiVRtSqAI/ct+mzaC1Oy/sApzlqMUY89N27PnH3pv61axfglK67o2PTQx3KzDKSue+SCr1Y88+P/bUVg1G3LGvbc06r+wwmQ91MpA3VXNWMqZeCmSSMxlfJZL2vYNNtvnJpcTlQlQSIcZ6mOM367a1unx9JJI1o+Oe+d1wFBTTtiMaCuLTMo8/MP91lgw66iuWQAAAAldEVYdGRhdGU6Y3JlYXRlADIwMjAtMDctMTlUMDM6Mzk6MjArMDA6MDCGZw5cAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDE5LTAxLTA4VDIwOjM0OjQzKzAwOjAwxXusFQAAACB0RVh0c29mdHdhcmUAaHR0cHM6Ly9pbWFnZW1hZ2ljay5vcme8zx2dAAAAGHRFWHRUaHVtYjo6RG9jdW1lbnQ6OlBhZ2VzADGn/7svAAAAGHRFWHRUaHVtYjo6SW1hZ2U6OkhlaWdodAAxNDgVJuYGAAAAF3RFWHRUaHVtYjo6SW1hZ2U6OldpZHRoADE0OIbXtlsAAAAZdEVYdFRodW1iOjpNaW1ldHlwZQBpbWFnZS9wbmc/slZOAAAAF3RFWHRUaHVtYjo6TVRpbWUAMTU0Njk3OTY4M2FiHTQAAAARdEVYdFRodW1iOjpTaXplADE1MDJCmyxecAAAAFp0RVh0VGh1bWI6OlVSSQBmaWxlOi8vL2RhdGEvd3d3cm9vdC93d3cuZWFzeWljb24ubmV0L2Nkbi1pbWcuZWFzeWljb24uY24vZmlsZXMvMTIyLzEyMjI2NDYucG5nk8ShNwAAAABJRU5ErkJggg=='
        self.BAR_MAX = 1000

    def create_layout(self):
        sg.theme('LightBrown12')
        layout = [
            # 菜单图标按钮
            [sg.Button('打开'), sg.Button('执行'), sg.Button('设置'), sg.Button('帮助')],
            # 显示视频预览
            [sg.Canvas(size=(854, 480), background_color='black')],
            # 播放暂停按钮 + 快进快退条
            [sg.Button('', size=(4, 1), image_data=self.play_pause_btn_base64),
             sg.Slider(size=(115, 20), range=(0, 475), key='-SLIDER-', orientation='h', enable_events=True,
                       disable_number_display=True)
             ],
            # 输出区域
            [sg.Output(size=(115, 10), font='Courier 10'),
             sg.Frame(title='字幕位置调整', layout=[[
                 sg.Slider(range=(1, 100), orientation='v', size=(10, 20), default_value=25),
                 sg.Slider(range=(1, 100), orientation='v', size=(10, 20), default_value=75),
             ]], pad=((15, 5), (0, 0)))],
            # 运行按钮 + 进度条
            [sg.Button('运行'), sg.ProgressBar(max_value=self.BAR_MAX, orientation='h', size=(90, 20), key='-PROG-')],
        ]

        window = sg.Window('Script launcher', layout)
        while True:
            event, values = window.read()
            if event == 'EXIT' or event == sg.WIN_CLOSED:
                break


if __name__ == '__main__':
    print(sg.theme_list())
    GUI().create_layout()
