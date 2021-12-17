# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao 
@Time    : 2021/3/24 9:28 上午
@FileName: main.py
@desc: 主程序入口文件
"""
import re
import os
import random
from collections import Counter
import unicodedata

import cv2
from Levenshtein import ratio
from PIL import Image
from numpy import average, dot, linalg
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(__file__))
import config
from tools.reformat import reformat
from tools.infer import utility
from tools.infer.predict_det import TextDetector
from tools.infer.predict_system import TextSystem
import platform


# 加载文本检测+识别模型
class OcrRecogniser:
    def __init__(self):
        # 获取参数对象
        self.args = utility.parse_args()
        self.recogniser = self.init_model()

    def predict(self, image):
        detection_box, recognise_result = self.recogniser(image)
        return detection_box, recognise_result

    def init_model(self):
        self.args.use_gpu = config.USE_GPU
        if config.USE_GPU:
            # 设置文本检测模型路径
            self.args.det_model_dir = config.DET_MODEL_PATH
            # 设置文本识别模型路径
            self.args.rec_model_dir = config.REC_MODEL_PATH
        else:
            # 加载快速模型
            self.args.det_model_dir = config.DET_MODEL_FAST_PATH
            # 加载快速模型
            self.args.rec_model_dir = config.REC_MODEL_FAST_PATH
        # 设置字典路径
        self.args.rec_char_dict_path = config.DICT_PATH
        # 设置识别文本的类型
        self.args.rec_char_type = config.REC_CHAR_TYPE
        return TextSystem(self.args)


class SubtitleDetect:
    def __init__(self):
        # 获取参数对象
        args = utility.parse_args()
        args.det_algorithm = 'DB'
        args.det_model_dir = config.DET_MODEL_FAST_PATH
        self.text_detector = TextDetector(args)

    def detect_subtitle(self, img):
        dt_boxes, elapse = self.text_detector(img)
        return dt_boxes, elapse


class SubtitleExtractor:
    """
    视频字幕提取类
    """

    def __init__(self, vd_path, sub_area=None):
        # 字幕区域位置
        self.sub_area = sub_area
        self.sub_detector = SubtitleDetect()
        # 临时存储文件夹
        self.temp_output_dir = os.path.join(os.path.dirname(config.BASE_DIR), 'output')
        # 视频路径
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(vd_path)
        # 视频帧总数
        self.frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # 视频帧率
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # 视频尺寸
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 字幕出现区域
        self.subtitle_area = config.SUBTITLE_AREA
        print(f'帧数：{self.frame_count}，帧率：{self.fps}')
        # 提取的视频帧储存目录
        self.frame_output_dir = os.path.join(self.temp_output_dir, 'frames')
        # 提取的字幕文件存储目录
        self.subtitle_output_dir = os.path.join(self.temp_output_dir, 'subtitle')
        # 定义vsf的字幕输出路径
        self.vsf_subtitle = os.path.join(self.subtitle_output_dir, 'raw_vsf.srt')
        # 不存在则创建文件夹
        if not os.path.exists(self.frame_output_dir):
            os.makedirs(self.frame_output_dir)
        if not os.path.exists(self.subtitle_output_dir):
            os.makedirs(self.subtitle_output_dir)
        # 提取的原始字幕文本存储路径
        self.raw_subtitle_path = os.path.join(self.subtitle_output_dir, 'raw.txt')
        # 自定义ocr对象
        self.ocr = OcrRecogniser()

    def run(self):
        """
        运行整个提取视频的步骤
        """
        if not self.__is_legal_video_path(self.video_path):
            print('【警告】程序运行中断！路径不合法！视频路径请不要带有空格！！！请修改路径名或者文件名后重新运行程序')
        else:
            print('【处理中】开启提取视频关键帧...')
            if self.sub_area is not None:
                # 如果开启精准模式
                if config.ACCURATE_MODE_ON:
                    # 开启精准模式并且有GPU加速
                    if config.USE_GPU:
                        self.extract_frame_by_det()
                    else:
                        self.extract_frame_by_fps()
                # 如果没有开启精准模式
                else:
                    # 没有开启精准模式且操作系统为Windows
                    if platform.system() == 'Windows':
                        self.extract_frame_by_vsf()
                    else:
                        self.extract_frame_by_fps()
            else:
                self.extract_frame_by_fps()
            print('【结束】提取视频关键帧完毕...')

            print('【处理中】开始提取字幕信息，此步骤可能花费较长时间，请耐心等待...')
            self.extract_subtitles()
            print('【结束】完成字幕提取，生成原始字幕文件...')

            if self.sub_area is None:
                print('【处理中】开始检测并过滤水印区域内容')
                # 询问用户视频是否有水印区域
                user_input = input('视频是否存在水印区域，存在的话输入y，不存在的话输入n: ').strip()
                if user_input == 'y':
                    self.filter_watermark()
                    print('【结束】已经成功过滤水印区域内容')
                else:
                    print('-----------------------------')

            if self.sub_area is None:
                print('【处理中】开始检测非字幕区域，并将非字幕区域的内容删除')
                self.filter_scene_text()
                print('【结束】已将非字幕区域的内容删除')
            # 如果识别的字幕语言包含英文，则将英文分词
            if config.REC_CHAR_TYPE in ('ch', 'EN', 'en'):
                reformat(os.path.join(os.path.splitext(self.video_path)[0] + '.srt'))
            print('【处理中】开始生成字幕文件')
            # 判断是否开启精准模式
            if config.ACCURATE_MODE_ON:
                # 如果开启精准模式则使用原生字幕生成
                self.generate_subtitle_file()
            else:
                # 如果没有开启精准模式，则Windows平台默认使用vsf提取
                if platform.system() == 'Windows':
                    self.generate_subtitle_file_vsf()
                else:
                    self.generate_subtitle_file()
            print('【结束】字幕文件生成成功')

    def extract_frame(self):
        """
        根据视频的分辨率，将高分辨的视频帧缩放到1280*720p
        根据字幕区域位置，将该图像区域截取出来
        """
        # 删除缓存
        self.__delete_frame_cache()

        # 当前视频帧的帧号
        frame_no = 0

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

                # 将当前帧与接下来的帧进行比较，计算余弦相似度
                compare_times = 0
                while self.video_cap.isOpened():
                    ret, frame_next = self.video_cap.read()
                    if ret:
                        frame_no += 1
                        frame_next = self._frame_preprocess(frame_next)
                        cosine_distance = self._compute_image_similarity(Image.fromarray(frame),
                                                                         Image.fromarray(frame_next))
                        compare_times += 1
                        if compare_times == config.FRAME_COMPARE_TIMES:
                            break
                        if cosine_distance > config.COSINE_SIMILARITY_THRESHOLD:
                            # 如果下一帧与当前帧的相似度大于设定阈值，则略过该帧
                            continue
                        # 如果相似度小于设定阈值，停止该while循环
                        else:
                            break
                    else:
                        break

        self.video_cap.release()

    def extract_frame_by_fps(self):
        """
        根据帧率，定时提取视频帧，容易丢字幕，但速度快
        """
        # 删除缓存
        self.__delete_frame_cache()

        # 当前视频帧的帧号
        frame_no = 0

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

    def extract_frame_by_det(self):
        """
        通过检测字幕区域位置提取字幕帧
        """
        # 删除缓存
        self.__delete_frame_cache()

        # 当前视频帧的帧号
        frame_no = 0

        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            # 如果读取视频帧失败（视频读到最后一帧）
            if not ret:
                break
            # 读取视频帧成功
            else:
                frame_no += 1
                if self.sub_area is not None:
                    ymin, ymax, xmin, xmax = self.sub_area
                    dt_boxes, elapse = self.sub_detector.detect_subtitle(frame[ymin:ymax, xmin:xmax])
                    if len(dt_boxes) > 0:
                        # 帧名往前补零，后续用于排序与时间戳转换，补足8位
                        # 一部10h电影，fps120帧最多也才1*60*60*120=432000 6位，所以8位足够
                        filename = os.path.join(self.frame_output_dir, str(frame_no).zfill(8) + '.jpg')
                        # 查询frame目录下最后两张图片
                        frame_list = sorted([i for i in os.listdir(self.frame_output_dir) if i.endswith('.jpg')])
                        # 如果frame列表大于等于2则取出最后两张图片
                        if len(frame_list) < 2:
                            # 保存视频帧
                            cv2.imwrite(filename, frame)
                        else:
                            frame_last = cv2.imread(os.path.join(self.frame_output_dir, frame_list[-1]))
                            frame_last_2nd = cv2.imread(os.path.join(self.frame_output_dir, frame_list[-2]))
                            if self._compare_ocr_result(frame_last, frame_last_2nd):
                                if self._compare_ocr_result(frame_last, frame):
                                    # 如果当最后两帧内容一样，且最后一帧与当前帧一样
                                    # 删除最后一张，将当前帧设置为最后一帧
                                    os.remove(os.path.join(self.frame_output_dir, frame_list[-1]))
                            cv2.imwrite(filename, frame)
                        print(f'字幕帧：{frame_no}, 耗时: {elapse}')

        self.video_cap.release()

    def extract_frame_by_vsf(self):
        """
       通过调用videoSubFinder获取字幕帧
       """
        # 删除缓存
        self.__delete_frame_cache()
        # 定义videoSubFinder所在路径
        path_vsf = os.path.join(config.BASE_DIR, '', 'subfinder', 'VideoSubFinderWXW.exe')
        # ：图像上半部分所占百分比，取值【0-1】
        top_end = 1 - self.sub_area[0] / self.frame_height
        # bottom_end：图像下半部分所占百分比，取值【0-1】
        bottom_end = 1 - self.sub_area[1] / self.frame_height
        # left_end：图像左半部分所占百分比，取值【0-1】
        left_end = self.sub_area[2] / self.frame_width
        # re：图像右半部分所占百分比，取值【0-1】
        right_end = self.sub_area[3] / self.frame_width
        # 定义执行命令
        cmd = path_vsf + " -c -r" + " -i \"" + self.video_path + "\" -o " + self.temp_output_dir + f' -ces {self.vsf_subtitle}' + f' -te {top_end}' + f' -be {bottom_end}' + f' -le {left_end}' + f' -re {right_end}'
        os.system(cmd)
        # 提取字幕帧
        cap = cv2.VideoCapture(self.video_path)
        for i, frame_name in enumerate(os.listdir(os.path.join(self.temp_output_dir, 'RGBImages'))):
            timestamp = frame_name.split('__')[0]
            h, m, s, ms = timestamp.split('_')
            total_ms = int(ms) + int(s) * 1000 + int(m) * 60 * 1000 + int(h) * 60 * 60 * 1000
            cap.set(cv2.CAP_PROP_POS_MSEC, total_ms)
            ret, frame = cap.read()
            if ret:
                img_name = os.path.join(self.frame_output_dir, f'{str(i + 1).zfill(8)}.jpg')
                cv2.imwrite(img_name, frame)
        # 释放占用资源
        cap.release()

    def extract_subtitle_frame(self):
        """
        提取包含字幕的视频帧
        """
        # 删除缓存
        self.__delete_frame_cache()
        # 获取字幕帧列表
        subtitle_frame_list = self._analyse_subtitle_frame()
        if subtitle_frame_list is None:
            print('请指定字幕区域')
            return
        cap = cv2.VideoCapture(self.video_path)
        idx = 0
        index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx in subtitle_frame_list and idx != 0:
                filename = os.path.join(self.frame_output_dir, str(idx).zfill(8) + '.jpg')
                frame = self._frame_preprocess(frame)
                cv2.imwrite(filename, frame)
                subtitle_frame_list.remove(idx)
                index += 1
            idx = idx + 1
        cap.release()

    def extract_subtitles(self):
        """
        提取视频帧中的字幕信息，生成一个txt文件
        """
        # 初始化文本识别对象
        text_recogniser = OcrRecogniser()
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
            dt_box, rec_res = text_recogniser.predict(img)
            # 获取文本坐标
            coordinates = self.__get_coordinates(dt_box)
            # 将结果写入txt文本中
            text_res = [(res[0], res[1]) for res in rec_res]
            for content, coordinate in zip(text_res, coordinates):
                if self.sub_area is not None:
                    s_ymin = self.sub_area[0]
                    s_ymax = self.sub_area[1]
                    s_xmin = self.sub_area[2]
                    s_xmax = self.sub_area[3]
                    xmin = coordinate[0]
                    xmax = coordinate[1]
                    ymin = coordinate[2]
                    ymax = coordinate[3]
                    if s_xmin <= xmin and xmax <= s_xmax and s_ymin <= ymin and ymax <= s_ymax:
                        print(content[0])
                        if content[1] > config.DROP_SCORE:
                            f.write(f'{os.path.splitext(frame)[0]}\t'
                                    f'{coordinate}\t'
                                    f'{content[0]}\n')
                else:
                    f.write(f'{os.path.splitext(frame)[0]}\t'
                            f'{coordinate}\t'
                            f'{content[0]}\n')
        # 关闭文件
        f.close()

    def filter_watermark(self):
        """
        去除原始字幕文本中的水印区域的文本
        """
        # 获取潜在水印区域
        watermark_areas = self._detect_watermark_area()

        # 从frame目录随机读取一张图片，将所水印区域标记出来，用户看图判断是否是水印区域
        frame_path = os.path.join(self.frame_output_dir,
                                  random.choice(
                                      [i for i in sorted(os.listdir(self.frame_output_dir)) if i.endswith('.jpg')]))
        sample_frame = cv2.imread(frame_path)

        # 给潜在的水印区域编号
        area_num = ['E', 'D', 'C', 'B', 'A']

        for watermark_area in watermark_areas:
            ymin = min(watermark_area[0][2], watermark_area[0][3])
            ymax = max(watermark_area[0][3], watermark_area[0][2])
            xmin = min(watermark_area[0][0], watermark_area[0][1])
            xmax = max(watermark_area[0][1], watermark_area[0][0])
            cover = sample_frame[ymin:ymax, xmin:xmax]
            cover = cv2.blur(cover, (10, 10))
            cv2.rectangle(cover, pt1=(0, cover.shape[0]), pt2=(cover.shape[1], 0), color=(0, 0, 255), thickness=3)
            sample_frame[ymin:ymax, xmin:xmax] = cover
            position = ((xmin + xmax) // 2, ymax)

            cv2.putText(sample_frame, text=area_num.pop(), org=position, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        sample_frame_file_path = os.path.join(os.path.dirname(self.frame_output_dir), 'watermark_area.jpg')
        cv2.imwrite(sample_frame_file_path, sample_frame)
        print(f'请查看图片, 确定水印区域: {sample_frame_file_path}')

        area_num = ['E', 'D', 'C', 'B', 'A']
        for watermark_area in watermark_areas:
            user_input = input(f'是否去除区域{area_num.pop()}{str(watermark_area)}中的字幕?'
                               f'\n输入 "y" 或 "回车" 表示去除，输入"n"或其他表示不去除: ').strip()
            if user_input == 'y' or user_input == '\n':
                with open(self.raw_subtitle_path, mode='r+', encoding='utf-8') as f:
                    content = f.readlines()
                    f.seek(0)
                    for i in content:
                        if i.find(str(watermark_area[0])) == -1:
                            f.write(i)
                    f.truncate()
                print(f'已经删除该区域字幕...')
        print('水印区域字幕过滤完毕...')
        # 删除缓存
        if os.path.exists(sample_frame_file_path):
            os.remove(sample_frame_file_path)

    def filter_scene_text(self):
        """
        将场景里提取的文字过滤，仅保留字幕区域
        """
        # 获取潜在字幕区域
        subtitle_area = self._detect_subtitle_area()[0][0]

        # 从frame目录随机读取一张图片，将所水印区域标记出来，用户看图判断是否是水印区域
        frame_path = os.path.join(self.frame_output_dir,
                                  random.choice(
                                      [i for i in sorted(os.listdir(self.frame_output_dir)) if i.endswith('.jpg')]))
        sample_frame = cv2.imread(frame_path)

        # 为了防止有双行字幕，根据容忍度，将字幕区域y范围加高
        ymin = abs(subtitle_area[0] - config.SUBTITLE_AREA_DEVIATION_PIXEL)
        ymax = subtitle_area[1] + config.SUBTITLE_AREA_DEVIATION_PIXEL
        # 画出字幕框的区域
        cv2.rectangle(sample_frame, pt1=(0, ymin), pt2=(sample_frame.shape[1], ymax), color=(0, 0, 255), thickness=3)
        sample_frame_file_path = os.path.join(os.path.dirname(self.frame_output_dir), 'subtitle_area.jpg')
        cv2.imwrite(sample_frame_file_path, sample_frame)
        print(f'请查看图片, 确定字幕区域是否正确: {sample_frame_file_path}')

        user_input = input(f'是否去除红色框区域外{(ymin, ymax)}的字幕?'
                           f'\n输入 "y" 或 "回车" 表示去除，输入"n"或其他表示不去除: ').strip()
        if user_input == 'y' or user_input == '\n':
            with open(self.raw_subtitle_path, mode='r+', encoding='utf-8') as f:
                content = f.readlines()
                f.seek(0)
                for i in content:
                    i_ymin = int(i.split('\t')[1].split('(')[1].split(')')[0].split(', ')[2])
                    i_ymax = int(i.split('\t')[1].split('(')[1].split(')')[0].split(', ')[3])
                    if ymin <= i_ymin and i_ymax <= ymax:
                        f.write(i)
                f.truncate()
            print('去除完毕')
        # 删除缓存
        if os.path.exists(sample_frame_file_path):
            os.remove(sample_frame_file_path)

    def generate_subtitle_file(self):
        """
        生成srt格式的字幕文件
        """
        subtitle_content = self._remove_duplicate_subtitle()
        srt_filename = os.path.join(os.path.splitext(self.video_path)[0] + '.srt')
        # 保存持续时间不足1秒的字幕行，用于后续处理
        post_process_subtitle = []
        with open(srt_filename, mode='w', encoding='utf-8') as f:
            for index, content in enumerate(subtitle_content):
                line_code = index + 1
                frame_start = self._frame_to_timecode(int(content[0]))
                # 比较起始帧号与结束帧号， 如果字幕持续时间不足1秒，则将显示时间设为1s
                if abs(int(content[1]) - int(content[0])) < self.fps:
                    frame_end = self._frame_to_timecode(int(int(content[0]) + self.fps))
                    post_process_subtitle.append(line_code)
                else:
                    frame_end = self._frame_to_timecode(int(content[1]))
                frame_content = content[2]
                subtitle_line = f'{line_code}\n{frame_start} --> {frame_end}\n{frame_content}\n'
                f.write(subtitle_line)
        print(f'字幕文件生成位置：{srt_filename}')
        # 返回持续时间低于1s的字幕行
        return post_process_subtitle

    def generate_subtitle_file_vsf(self):
        try:
            subtitle_timestamp = []
            with open(self.vsf_subtitle, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
                timestamp = []
                frame_no = []
                for line in lines:
                    if re.match(r'^\d{1,}$', line):
                        frame_no.append(line.replace('\n', '').replace('\r', '').zfill(8))
                    if re.match(r'^\d{2,}:\d{2,}:\d{2,},\d{1,3}.*', line):
                        timestamp.append(line.replace('\n', '').replace('\r', ''))
                for i in zip(frame_no, timestamp):
                    subtitle_timestamp.append(i)
            subtitle_content = self._remove_duplicate_subtitle()
            final_subtitle = []
            for sc in subtitle_content:
                frame_no = sc[0]
                content = sc[2]
                for st in subtitle_timestamp:
                    if st[0] == frame_no:
                        timestamp = st[1]
                        final_subtitle.append((timestamp, content))
            srt_filename = os.path.join(os.path.splitext(self.video_path)[0] + '.srt')
            with open(srt_filename, mode='w', encoding='utf-8') as f:
                for i, subtitle_line in enumerate(final_subtitle):
                    f.write(f'{i + 1}\n')
                    f.write(f'{subtitle_line[0]}\n')
                    f.write(f'{subtitle_line[1]}\n')
            print(f'字幕文件生成位置：{srt_filename}')
        except FileNotFoundError:
            self.generate_subtitle_file()

    def _analyse_subtitle_frame(self):
        """
        使用简单的图像算法找出包含字幕的视频帧
        : 参考 https://github.com/BruceHan98/OCR-Extract-Subtitles/blob/main/analyze_key_frame.py
        """
        if self.sub_area is None:
            return None
        else:
            subtitle_frame_index_list = []
            index = 0
            s_ymin = self.sub_area[0]
            s_ymax = self.sub_area[1]
            s_xmin = self.sub_area[2]
            s_xmax = self.sub_area[3]
            cap = cv2.VideoCapture(self.video_path)
            success, frame = cap.read()
            if success:
                # 截取字幕部分
                frame = frame[s_ymin:s_ymax, s_xmin:s_xmax]
            h, w = frame.shape[0:2]
            if config.BG_MOD == config.BackgroundColor.DARK:  # 深色背景
                minuend = np.full(h * w, config.BG_VALUE_DARK)  # 被减矩阵
            else:
                minuend = np.full(h * w, config.BG_VALUE_OTHER)  # 被减矩阵

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flatten_gray = gray.flatten()
            last_roi = flatten_gray - minuend
            last_roi = np.where(last_roi > 0, 1, 0)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = frame[s_ymin:s_ymax, s_xmin:s_xmax]
                if index % config.EXTRACT_INTERVAL == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    flatten_gray = gray.flatten()
                    roi = flatten_gray - minuend
                    roi = np.where(roi > 0, 1, 0)
                    change = roi - last_roi
                    addi = np.where(change > 0, 1, 0).sum()
                    if addi > roi.sum() * config.ROI_RATE:  # 字幕增加
                        subtitle_frame_index_list.append(index)
                    last_roi = roi
                index += 1

            cap.release()
            return subtitle_frame_index_list

    def _detect_watermark_area(self):
        """
        根据识别出来的raw txt文件中的坐标点信息，查找水印区域
        假定：水印区域（台标）的坐标在水平和垂直方向都是固定的，也就是具有(xmin, xmax, ymin, ymax)相对固定
        根据坐标点信息，进行统计，将一直具有固定坐标的文本区域选出
        :return 返回最有可能的水印区域
        """
        f = open(self.raw_subtitle_path, mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8’编码读取
        line = f.readline()  # 以行的形式进行读取文件
        # 坐标点列表
        coordinates_list = []
        # 帧列表
        frame_no_list = []
        # 内容列表
        content_list = []
        while line:
            frame_no = line.split('\t')[0]
            text_position = line.split('\t')[1].split('(')[1].split(')')[0].split(', ')
            content = line.split('\t')[2]
            frame_no_list.append(frame_no)
            coordinates_list.append((int(text_position[0]),
                                     int(text_position[1]),
                                     int(text_position[2]),
                                     int(text_position[3])))
            content_list.append(content)
            line = f.readline()
        f.close()
        # 将坐标列表的相似值统一
        coordinates_list = self._unite_coordinates(coordinates_list)

        # 将原txt文件的坐标更新为归一后的坐标
        with open(self.raw_subtitle_path, mode='w', encoding='utf-8') as f:
            for frame_no, coordinate, content in zip(frame_no_list, coordinates_list, content_list):
                f.write(f'{frame_no}\t{coordinate}\t{content}')

        if len(Counter(coordinates_list).most_common()) > config.WATERMARK_AREA_NUM:
            # 读取配置文件，返回可能为水印区域的坐标列表
            return Counter(coordinates_list).most_common(config.WATERMARK_AREA_NUM)
        else:
            # 不够则有几个返回几个
            return Counter(coordinates_list).most_common()

    def _detect_subtitle_area(self):
        """
        读取过滤水印区域后的raw txt文件，根据坐标信息，查找字幕区域
        假定：字幕区域在y轴上有一个相对固定的坐标范围，相对于场景文本，这个范围出现频率更高
        :return 返回字幕的区域位置
        """
        # 打开去水印区域处理过的raw txt
        f = open(self.raw_subtitle_path, mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8’编码读取
        line = f.readline()  # 以行的形式进行读取文件
        # y坐标点列表
        y_coordinates_list = []
        while line:
            text_position = line.split('\t')[1].split('(')[1].split(')')[0].split(', ')
            y_coordinates_list.append((int(text_position[2]), int(text_position[3])))
            line = f.readline()
        f.close()
        return Counter(y_coordinates_list).most_common(1)

    def _frame_preprocess(self, frame):
        """
        将视频帧进行裁剪
        """
        # 对于分辨率大于1920*1080的视频，将其视频帧进行等比缩放至1280*720进行识别
        # paddlepaddle会将图像压缩为640*640
        # if self.frame_width > 1280:
        #     scale_rate = round(float(1280 / self.frame_width), 2)
        #     frames = cv2.resize(frames, None, fx=scale_rate, fy=scale_rate, interpolation=cv2.INTER_AREA)
        cropped = int(frame.shape[0] // 2)
        # 如果字幕出现的区域在下部分
        if self.subtitle_area == config.SubtitleArea.LOWER_PART:
            # 将视频帧切割为下半部分
            frame = frame[cropped:]
        # 如果字幕出现的区域在上半部分
        elif self.subtitle_area == config.SubtitleArea.UPPER_PART:
            # 将视频帧切割为下半部分
            frame = frame[:cropped]
        return frame

    def _frame_to_timecode(self, frame_no):
        """
        将视频帧转换成时间
        :param frame_no: 视频的帧号，i.e. 第几帧视频帧
        :returns: SMPTE格式时间戳 as string, 如'01:02:12:32' 或者 '01:02:12;32'
        """
        # 设置当前帧号
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        cap.read()
        # 获取当前帧号对应的时间戳
        milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
        seconds = milliseconds // 1000
        milliseconds = int(milliseconds % 1000)
        minutes = 0
        hours = 0
        if seconds >= 60:
            minutes = int(seconds // 60)
            seconds = int(seconds % 60)
        if minutes >= 60:
            hours = int(minutes // 60)
            minutes = int(minutes % 60)
        smpte_token = ','
        cap.release()
        return "%02d:%02d:%02d%s%02d" % (hours, minutes, seconds, smpte_token, milliseconds)

    def _remove_duplicate_subtitle(self):
        """
        读取原始的raw txt，去除重复行，返回去除了重复后的字幕列表
        """
        self._concat_content_with_same_frameno()
        with open(self.raw_subtitle_path, mode='r', encoding='utf-8') as r:
            lines = r.readlines()
        content_list = []
        for line in lines:
            frame_no = line.split('\t')[0]
            content = line.split('\t')[2]
            content_list.append((frame_no, content))
        # 循环遍历每行字幕，记录开始时间与结束时间
        index = 0
        # 去重后的字幕列表
        unique_subtitle_list = []
        for i in content_list:
            # TODO: 时间复杂度非常高，有待优化
            # 定义字幕开始帧帧号
            start_frame = i[0]
            for j in content_list[index:]:
                # 计算当前行与下一行的Levenshtein距离
                distance = ratio(i[1], j[1])
                if distance < config.THRESHOLD_TEXT_SIMILARITY or j == content_list[-1]:
                    # 定义字幕结束帧帧号
                    end_frame = content_list[content_list.index(j) - 1][0]
                    if end_frame == start_frame:
                        end_frame = j[0]
                    # 如果是第一行字幕，直接添加进列表
                    if len(unique_subtitle_list) < 1:
                        unique_subtitle_list.append((start_frame, end_frame, i[1]))
                    else:
                        string_a = unique_subtitle_list[-1][2].replace(' ', '')
                        string_b = i[1].replace(' ', '')
                        similarity_ratio = ratio(string_a, string_b)
                        # 打印相似度
                        # print(f'{similarity_ratio}: {unique_subtitle_list[-1][2]} vs {i[1]}')
                        # 如果相似度小于阈值，说明该两行字幕不一样
                        if similarity_ratio < config.THRESHOLD_TEXT_SIMILARITY:
                            unique_subtitle_list.append((start_frame, end_frame, i[1]))
                        else:
                            # 如果大于阈值，但又不完全相同，说明两行字幕相似
                            # 可能出现以下情况: "但如何进人并接管上海" vs "但如何进入并接管上海"
                            # OCR识别出现了错误识别
                            if similarity_ratio < 1:
                                # TODO:
                                # 1) 取出两行字幕的并集
                                # 2) 纠错
                                # print(f'{round(similarity_ratio, 2)}, 需要手动纠错:\n {string_a} vs\n {string_b}')
                                # 保存较长的
                                if len(string_a) < len(string_b):
                                    unique_subtitle_list[-1] = (start_frame, end_frame, i[1])
                    index += 1
                    break
                else:
                    continue
        return unique_subtitle_list

    def _concat_content_with_same_frameno(self):
        """
        将raw txt文本中具有相同帧号的字幕行合并
        """
        with open(self.raw_subtitle_path, mode='r', encoding='utf-8') as r:
            lines = r.readlines()
        content_list = []
        frame_no_list = []
        for line in lines:
            frame_no = line.split('\t')[0]
            frame_no_list.append(frame_no)
            coordinate = line.split('\t')[1]
            content = line.split('\t')[2]
            content_list.append([frame_no, coordinate, content])

        # 找出那些不止一行的帧号
        frame_no_list = [i[0] for i in Counter(frame_no_list).most_common() if i[1] > 1]

        # 找出这些帧号出现的位置
        concatenation_list = []
        for frame_no in frame_no_list:
            position = [i for i, x in enumerate(content_list) if x[0] == frame_no]
            concatenation_list.append((frame_no, position))

        for i in concatenation_list:
            content = []
            for j in i[1]:
                content.append(content_list[j][2])
            content = ' '.join(content).replace('\n', ' ') + '\n'
            for k in i[1]:
                content_list[k][2] = content

        # 将多余的字幕行删除
        to_delete = []
        for i in concatenation_list:
            for j in i[1][1:]:
                to_delete.append(content_list[j])
        for i in to_delete:
            if i in content_list:
                content_list.remove(i)

        with open(self.raw_subtitle_path, mode='w', encoding='utf-8') as f:
            for frame_no, coordinate, content in content_list:
                content = unicodedata.normalize('NFKC', content)
                f.write(f'{frame_no}\t{coordinate}\t{content}')

    def _unite_coordinates(self, coordinates_list):
        """
        给定一个坐标列表，将这个列表中相似的坐标统一为一个值
        e.g. 由于检测框检测的结果不是一致的，相同位置文字的坐标可能一次检测为(255,123,456,789)，另一次检测为(253,122,456,799)
        因此要对相似的坐标进行值的统一
        :param coordinates_list 包含坐标点的列表
        :return: 返回一个统一值后的坐标列表
        """
        # 将相似的坐标统一为一个
        index = 0
        for coordinate in coordinates_list:  # TODO：时间复杂度n^2，待优化
            for i in coordinates_list:
                if self.__is_coordinate_similar(coordinate, i):
                    coordinates_list[index] = i
            index += 1
        return coordinates_list

    def _compute_image_similarity(self, image1, image2):
        """
        计算两张图片的余弦相似度
        """
        image1 = self.__get_thum(image1)
        image2 = self.__get_thum(image2)
        images = [image1, image2]
        vectors = []
        norms = []
        for image in images:
            vector = []
            for pixel_tuple in image.getdata():
                vector.append(average(pixel_tuple))
            vectors.append(vector)
            # linalg=linear（线性）+algebra（代数），norm则表示范数
            # 求图片的范数
            norms.append(linalg.norm(vector, 2))
        a, b = vectors
        a_norm, b_norm = norms
        # dot返回的是点积，对二维数组（矩阵）进行计算
        res = dot(a / a_norm, b / b_norm)
        return res

    def __get_area_text(self, ocr_result):
        """
        获取字幕区域内的文本内容
        """
        box, text = ocr_result
        coordinates = self.__get_coordinates(box)
        area_text = []
        for content, coordinate in zip(text, coordinates):
            if self.sub_area is not None:
                s_ymin = self.sub_area[0]
                s_ymax = self.sub_area[1]
                s_xmin = self.sub_area[2]
                s_xmax = self.sub_area[3]
                xmin = coordinate[0]
                xmax = coordinate[1]
                ymin = coordinate[2]
                ymax = coordinate[3]
                if s_xmin <= xmin and xmax <= s_xmax and s_ymin <= ymin and ymax <= s_ymax:
                    area_text.append(content[0])
        return area_text

    def _compare_ocr_result(self, img1, img2):
        """
        比较两张图片预测出的字幕区域文本是否相同
        """
        area_text1 = "".join(self.__get_area_text(self.ocr.predict(img1)))
        area_text2 = "".join(self.__get_area_text(self.ocr.predict(img2)))
        if ratio(area_text1, area_text2) > config.THRESHOLD_TEXT_SIMILARITY:
            return True
        else:
            return False

    @staticmethod
    def __is_legal_video_path(video_path):
        """
        判断视频路径是否合法， 如果不合法则修改路径
        """
        is_legal = True
        # 是否包含空格
        if re.search(r"\s", video_path):
            is_legal = False
        return is_legal

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

    @staticmethod
    def __is_coordinate_similar(coordinate1, coordinate2):
        """
        计算两个坐标是否相似，如果两个坐标点的xmin,xmax,ymin,ymax的差值都在像素点容忍度内
        则认为这两个坐标点相似
        """
        return abs(coordinate1[0] - coordinate2[0]) < config.PIXEL_TOLERANCE_X and \
               abs(coordinate1[1] - coordinate2[1]) < config.PIXEL_TOLERANCE_X and \
               abs(coordinate1[2] - coordinate2[2]) < config.PIXEL_TOLERANCE_Y and \
               abs(coordinate1[3] - coordinate2[3]) < config.PIXEL_TOLERANCE_Y

    @staticmethod
    def __get_thum(image, size=(64, 64), greyscale=False):
        """
        对图片进行统一化处理
        """
        # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
        image = image.resize(size, Image.ANTIALIAS)
        if greyscale:
            # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
            image = image.convert('L')
        return image

    def __delete_frame_cache(self):
        if len(os.listdir(self.frame_output_dir)) > 0:
            for i in os.listdir(self.frame_output_dir):
                os.remove(os.path.join(self.frame_output_dir, i))


if __name__ == '__main__':
    # 提示用户输入视频路径
    video_path = input("请输入视频完整路径：").strip()
    # 提示用户输入字幕区域
    ymin, ymax, xmin, xmax = map(int, input("请输入字幕区域(ymin ymax xmin xmax)：").split())
    subtitle_area = (ymin, ymax, xmin, xmax)
    # 新建字幕提取对象
    se = SubtitleExtractor(video_path, subtitle_area)
    # 开始提取字幕
    se.run()
