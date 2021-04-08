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

    def create_layout(self):
        sg.theme('DarkAmber')
        layout = [
            [sg.Graph(canvas_size=(720, 405), graph_bottom_left=(-105, -105), graph_top_right=(105, 105),
                      background_color='white', key='graph')],
            [sg.Button('', image_data=self.play_pause_btn_base64, border_width=0),
             sg.Slider((0, 405), size=(88, 20), key='-SLIDER-', orientation='h', enable_events=True,
                       disable_number_display=True)],
            [sg.Output(size=(88, 20), font='Courier 10')],
            [sg.Button('script1'), sg.Button('script2'), sg.Button('EXIT')],
            [sg.Text('Manual command', size=(15, 1)), sg.Input(focus=True, key='-IN-'),
             sg.Button('Run', bind_return_key=True), sg.Button('Run No Wait')]
        ]
        window = sg.Window('Script launcher', layout)
        while True:
            event, values = window.read()
            if event == 'EXIT' or event == sg.WIN_CLOSED:
                break  # exit button clicked
            if event == 'script1':
                sp = sg.execute_command_subprocess('pip', 'list', wait=True)
                print(sg.execute_get_results(sp)[0])
            elif event == 'script2':
                print(f'Running python --version')
                # For this one we need to wait for the subprocess to complete to get the results
                sp = sg.execute_command_subprocess('python', '--version', wait=True)
                print(sg.execute_get_results(sp)[0])
            elif event == 'Run':
                args = values['-IN-'].split(' ')
                print(f'Running {values["-IN-"]} args={args}')
                sp = sg.execute_command_subprocess(args[0], *args[1:])
                # This will cause the program to wait for the subprocess to finish
                print(sg.execute_get_results(sp)[0])
            elif event == 'Run No Wait':
                args = values['-IN-'].split(' ')
                print(f'Running {values["-IN-"]} args={args}', 'Results will not be shown')
                sp = sg.execute_command_subprocess(args[0], *args[1:])


if __name__ == '__main__':
    print(sg.theme_list())
    GUI().create_layout()
