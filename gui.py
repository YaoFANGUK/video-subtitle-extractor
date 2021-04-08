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
from tqdm import tqdm
import config
from threading import Thread, Lock
from backend.tools.infer import utility
from backend.tools.infer.predict_system import TextSystem


# 加载文本检测+识别模型
def load_model():
    # 获取参数对象
    args = utility.parse_args()
    # 设置文本检测模型路径
    # args.det_model_dir = config.DET_MODEL_PATH
    # 加载快速模型
    args.det_model_dir = config.DET_MODEL_FAST_PATH
    # 设置文本识别模型路径
    # args.rec_model_dir = config.REC_MODEL_PATH
    # 加载快速模型
    args.rec_model_dir = config.REC_MODEL_FAST_PATH
    # 设置字典路径
    args.rec_char_dict_path = config.DICT_PATH
    # 是否使用GPU加速
    args.use_gpu = config.USE_GPU
    return TextSystem(args)


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
        # 定义线程锁
        self.lock = Lock()  # 获得线程锁
        # 进度条最大值
        self.BAR_MAX = 1000
        # 临时存储文件夹
        self.temp_output_dir = os.path.join(config.BASE_DIR, 'output')
        # 提取的视频帧储存目录
        self.frame_output_dir = os.path.join(self.temp_output_dir, 'frames')
        # 提取的字幕文件存储目录
        self.subtitle_output_dir = os.path.join(self.temp_output_dir, 'subtitle')
        # 不存在则创建文件夹
        if not os.path.exists(self.frame_output_dir):
            os.makedirs(self.frame_output_dir)
        if not os.path.exists(self.subtitle_output_dir):
            os.makedirs(self.subtitle_output_dir)
        # 提取的原始字幕文本存储路径
        self.raw_subtitle_path = os.path.join(self.subtitle_output_dir, 'raw.txt')

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
            [sg.Slider(size=(95, 20), range=(1, 1), key='-SLIDER-', orientation='h', enable_events=True,
                       disable_number_display=True)
             ],
            # 输出区域
            [sg.Output(size=(70, 10), font='Courier 10'),
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
                    # 获取视频的帧率
                    self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
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
                self.xmin = int(values['-X-SLIDER-'])
                self.xmax = int(values['-X-SLIDER-'] + values['-X-SLIDER-W-'])
                self.ymin = int(values['-Y-SLIDER-'])
                self.ymax = int(values['-Y-SLIDER-'] + values['-Y-SLIDER-H-'])
                print(f'字幕区域：({self.ymin},{self.ymax},{self.xmin},{self.xmax})')

            # 【Step 1】提取字幕帧
            Thread(target=self.extract_frame_by_fps).start()
            # 【Step 2】提取字幕信息
            Thread(target=self.extract_subtitles).start()

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

    def extract_frame_by_fps(self):
        """
        根据帧率，定时提取视频帧，容易丢字幕，但速度快
        """
        # 加锁
        lock.aquire()
        print('【处理中】开始提取字幕帧...')
        # 删除缓存
        self.__delete_frame_cache()

        # 当前视频帧的帧号
        frame_no = 0
        # 从第一帧开始读取
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            # 如果读取视频帧失败（视频读到最后一帧）
            if not ret:
                break
            # 读取视频帧成功
            else:
                frame_no += 1
                frame = self._frame_preprocess(frame)

                # 帧名往前补零，后续用于排序与时间戳转换，补足8位
                # 一部10h电影，fps120帧最多也才1*60*60*120=432000 6位，所以8位足够
                filename = os.path.join(self.frame_output_dir, str(frame_no).zfill(8) + '.jpg')
                # 保存视频帧
                cv2.imwrite(filename, frame)

                # 跳过剩下的帧
                for i in range(int(self.fps // config.EXTRACT_FREQUENCY) - 1):
                    ret, _ = self.video_cap.read()
                    if ret:
                        frame_no += 1
        self.video_cap.release()
        lock.release()  # 释放锁
        print('【结束】字幕帧提取完毕')

    def extract_subtitles(self):
        """
        提取视频帧中的字幕信息，生成一个txt文件
        """
        print('【处理中】开始提取字幕信息，此步骤可能花费较长时间，请耐心等待...')
        # 初始化文本识别对象
        text_recogniser = load_model()
        # 视频帧列表
        frame_list = [i for i in sorted(os.listdir(self.frame_output_dir)) if i.endswith('.jpg')]
        # 删除缓存
        if os.path.exists(self.raw_subtitle_path):
            os.remove(self.raw_subtitle_path)
        # 新建文件
        f = open(self.raw_subtitle_path, mode='w+', encoding='utf-8')

        for frame in frame_list:
            # 读取视频帧
            img = cv2.imread(os.path.join(self.frame_output_dir, frame))
            # 获取检测结果
            dt_box, rec_res = text_recogniser(img)
            # 获取文本坐标
            coordinates = self.__get_coordinates(dt_box)
            # 将结果写入txt文本中
            text_res = [res[0] for res in rec_res]
            if len(text_res) > 1:
                print(text_res)
            for content, coordinate in zip(text_res, coordinates):
                f.write(f'{os.path.splitext(frame)[0]}\t'
                        f'{coordinate}\t'
                        f'{content}\n')
        # 关闭文件
        f.close()
        print('【结束】字幕提取完毕，生成原始字幕文件...')

    def _frame_preprocess(self, frame):
        """
        将视频帧进行裁剪
        """
        return frame[self.ymin:self.ymax, self.xmin:self.xmax]

    def __delete_frame_cache(self):
        if len(os.listdir(self.frame_output_dir)) > 0:
            for i in os.listdir(self.frame_output_dir):
                os.remove(os.path.join(self.frame_output_dir, i))

    @staticmethod
    def __get_coordinates(dt_box):
        """
        从返回的检测框中获取坐标
        :param dt_box 检测框返回结果
        :return list 坐标点列表
        """
        coordinate_list = list()
        if isinstance(dt_box, list):
            for i in dt_box:
                i = list(i)
                (x1, y1) = int(i[0][0]), int(i[0][1])
                (x2, y2) = int(i[1][0]), int(i[1][1])
                (x3, y3) = int(i[2][0]), int(i[2][1])
                (x4, y4) = int(i[3][0]), int(i[3][1])
                xmin = max(x1, x4)
                xmax = min(x2, x3)
                ymin = max(y1, y2)
                ymax = min(y3, y4)
                coordinate_list.append((xmin, xmax, ymin, ymax))
        return coordinate_list


if __name__ == '__main__':
    # 创建GUI对象
    subtitleExtractorGUI = SubtitleExtractorGUI()
    # 运行字幕提取器
    subtitleExtractorGUI.run()
