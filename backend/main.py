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
import shutil
from collections import Counter
import unicodedata
from threading import Thread
from pathlib import Path
import cv2
from Levenshtein import ratio
from PIL import Image
from numpy import average, dot, linalg
import numpy as np
from tqdm import tqdm
import sys
from paddle import fluid
fluid.install_check.run_check()
sys.path.insert(0, os.path.dirname(__file__))
import config
from config import interface_config
from tools.reformat_en import reformat
from tools.infer import utility
from tools.infer.predict_det import TextDetector
from tools.ocr import OcrRecogniser, get_coordinates
from tools import subtitle_ocr
from tools.constant import BackgroundColor
import threading
import platform
import multiprocessing
import time


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
        # 线程锁
        self.lock = threading.RLock()
        # 字幕区域位置
        self.sub_area = sub_area
        self.sub_detector = SubtitleDetect()
        # 视频路径
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(vd_path)
        # 临时存储文件夹
        self.vd_name = Path(self.video_path).stem
        self.temp_output_dir = os.path.join(os.path.dirname(config.BASE_DIR), 'output', str(self.vd_name))
        # 视频帧总数
        self.frame_count = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # 视频帧率
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # 视频尺寸
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 字幕出现区域
        self.subtitle_area = config.SUBTITLE_AREA
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
        print(f"{interface_config['Main']['RecSubLang']}：{config.REC_CHAR_TYPE}")
        print(f"{interface_config['Main']['RecMode']}：{config.MODE_TYPE}")
        if config.USE_GPU:
            print(interface_config['Main']['GPUSpeedUp'])
        # 处理进度
        self.progress = 0
        # 是否完成
        self.isFinished = False
        # ocr队列
        self.subtitle_ocr_queue = None
        # vsf运行状态
        self.vsf_running = False

    def run(self):
        """
        运行整个提取视频的步骤
        """
        start_time = time.time()
        self.lock.acquire()
        print(f"{interface_config['Main']['FrameCount']}：{self.frame_count}，{interface_config['Main']['FrameRate']}：{self.fps}")
        print(interface_config['Main']['StartProcessFrame'])

        subtitle_ocr_thread, self.subtitle_ocr_queue = subtitle_ocr.async_start(self.video_path, self.raw_subtitle_path, self.sub_area,
                                                                                config.REC_CHAR_TYPE, config.DROP_SCORE)
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

        duration_ms = (self.frame_count / self.fps) * 1000
        self.subtitle_ocr_queue.put((-1, duration_ms, -1, None, None, None))
        subtitle_ocr_thread.join()
        print(interface_config['Main']['FinishProcessFrame'])

        print(interface_config['Main']['StartFindSub'])
        # 重置进度条
        self.progress = 0
        print(interface_config['Main']['FinishFindSub'])

        if self.sub_area is None:
            print(interface_config['Main']['StartDetectWaterMark'])
            # 询问用户视频是否有水印区域
            user_input = input(interface_config['Main']['checkWaterMark']).strip()
            if user_input == 'y':
                self.filter_watermark()
                print(interface_config['Main']['FinishDetectWaterMark'])
            else:
                print('-----------------------------')

        if self.sub_area is None:
            print(interface_config['Main']['StartDeleteNonSub'])
            self.filter_scene_text()
            print(interface_config['Main']['FinishDeleteNonSub'])
        print(interface_config['Main']['StartGenerateSub'])
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
        # 如果识别的字幕语言包含英文，则将英文分词
        if config.REC_CHAR_TYPE in ('ch', 'EN', 'en', 'ch_tra'):
            reformat(os.path.join(os.path.splitext(self.video_path)[0] + '.srt'))
        print(interface_config['Main']['FinishGenerateSub'], f"{round(time.time() - start_time, 2)}s")
        self.progress = 100
        self.isFinished = True
        # 删除缓存文件
        self.empty_cache()
        self.lock.release()

    def extract_frame_by_fps(self):
        """
        根据帧率，定时提取视频帧，容易丢字幕，但速度快
        """
        # 删除缓存
        self.__delete_frame_cache()

        # 当前视频帧的帧号
        frame_no = 0

        duration_ms = (self.frame_count / self.fps) * 1000
        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            # 如果读取视频帧失败（视频读到最后一帧）
            if not ret:
                break
            # 读取视频帧成功
            else:
                total_ms = self.video_cap.get(cv2.CAP_PROP_POS_MSEC)
                if total_ms <= 0:
                    total_ms = self._frameno_to_milliseconds(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
                frame_no += 1
                self.subtitle_ocr_queue.put((total_ms, duration_ms, frame_no, None, None, self.subtitle_area))
                # 跳过剩下的帧
                for i in range(int(self.fps // config.EXTRACT_FREQUENCY) - 1):
                    ret, _ = self.video_cap.read()
                    if ret:
                        frame_no += 1
                        # 更新进度条
                        self.progress = (frame_no / self.frame_count) * 100

        self.video_cap.release()

    def extract_frame_by_det(self):
        """
        通过检测字幕区域位置提取字幕帧
        """
        # 删除缓存
        self.__delete_frame_cache()

        # 当前视频帧的帧号
        frame_no = 0
        frame_lru_list = []
        frame_lru_list_max_size = 2
        ocr_args_list = []
        duration_ms = (self.frame_count / self.fps) * 1000
        compare_ocr_result_cache = {}
        tbar = tqdm(total=int(self.frame_count), unit='f', position=0, file=sys.__stdout__)
        while self.video_cap.isOpened():
            ret, frame = self.video_cap.read()
            # 如果读取视频帧失败（视频读到最后一帧）
            if not ret:
                break
            total_ms = self.video_cap.get(cv2.CAP_PROP_POS_MSEC)
            if total_ms <= 0:
                total_ms = self._frameno_to_milliseconds(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
            # 读取视频帧成功
            frame_no += 1
            dt_boxes, elapse = self.sub_detector.detect_subtitle(frame)
            has_subtitle = False
            if self.sub_area is not None:
                s_ymin, s_ymax, s_xmin, s_xmax = self.sub_area
                for box in dt_boxes:
                    xmin, xmax, ymin, ymax = box[0], box[1], box[2], box[3]
                    if (s_xmin <= xmin).any() and (xmax <= s_xmax).any() \
                            and (s_ymin <= ymin).any() and (ymax <= s_ymax).any():
                        has_subtitle = True
                        break
            else:
                has_subtitle = len(dt_boxes) > 0
            if has_subtitle:
                # 如果frame列表大于等于2则取出最后两张图片
                if len(frame_lru_list) >= 2:
                    frame_last, frame_last_no = frame_lru_list[-1]
                    frame_last_2nd, frame_last_2nd_no = frame_lru_list[-2]
                    if self._compare_ocr_result(compare_ocr_result_cache,
                                                frame_last, frame_last_no,
                                                frame_last_2nd, frame_last_2nd_no):
                        if self._compare_ocr_result(compare_ocr_result_cache,
                                                    frame_last, frame_last_no,
                                                    frame, frame_no):
                            # 如果当最后两帧内容一样，且最后一帧与当前帧一样
                            # 删除最后一张，将当前帧设置为最后一帧
                            ocr_args_list.pop(-1)
                            frame_lru_list.pop(-1)
                frame_lru_list.append((frame, frame_no))
                ocr_args_list.append((total_ms, duration_ms, frame_no))

                while len(frame_lru_list) > frame_lru_list_max_size:
                    frame_lru_list.pop(0)
                while len(ocr_args_list) > 1:
                    ocr_info_total_ms, ocr_info_duration_ms, ocr_info_frame_no = ocr_args_list.pop(0)
                    if frame_no in compare_ocr_result_cache:
                        predict_result = compare_ocr_result_cache[frame_no]
                        dt_box, rec_res = predict_result['dt_box'], predict_result['rec_res']
                    else:
                        dt_box, rec_res = None, None
                    self.subtitle_ocr_queue.put((ocr_info_total_ms, ocr_info_duration_ms, ocr_info_frame_no,
                                                 dt_box, rec_res, self.subtitle_area))
                self.progress = (frame_no / self.frame_count) * 100
            tbar.update(1)
        while len(ocr_args_list) > 0:
            ocr_info_total_ms, ocr_info_duration_ms, ocr_info_frame_no = ocr_args_list.pop(0)
            if frame_no in compare_ocr_result_cache:
                predict_result = compare_ocr_result_cache[frame_no]
                dt_box, rec_res = predict_result['dt_box'], predict_result['rec_res']
            else:
                dt_box, rec_res = None, None
            self.subtitle_ocr_queue.put((ocr_info_total_ms, ocr_info_duration_ms, ocr_info_frame_no,
                                         dt_box, rec_res, self.subtitle_area))
        self.video_cap.release()

    def extract_frame_by_vsf(self):
        """
       通过调用videoSubFinder获取字幕帧
       """
        def count_process():
            duration_ms = (self.frame_count / self.fps) * 1000
            last_total_ms = 0
            processed_image = set()
            rgb_images_path = os.path.join(self.temp_output_dir, 'RGBImages')
            while self.vsf_running and not self.isFinished:
                if not os.path.exists(rgb_images_path):
                    continue
                try:
                    rgb_images = sorted(os.listdir(rgb_images_path))
                    for rgb_image in rgb_images:
                        if rgb_image in processed_image:
                            continue
                        processed_image.add(rgb_image)
                        h, m, s, ms = rgb_image.split('__')[0].split('_')
                        total_ms = int(ms) + int(s) * 1000 + int(m) * 60 * 1000 + int(h) * 60 * 60 * 1000
                        if total_ms > last_total_ms:
                            self.subtitle_ocr_queue.put((total_ms, duration_ms, -1, None, None, self.subtitle_area))
                        last_total_ms = total_ms
                        if total_ms / duration_ms >= 1:
                            self.progress = 100
                            return
                        else:
                            self.progress = (total_ms / duration_ms) * 100
                # 文件被清理了
                except FileNotFoundError:
                    return

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
        cpu_count = max(int(multiprocessing.cpu_count() * 2 / 3), 1)
        if cpu_count < 4:
            cpu_count = max(multiprocessing.cpu_count() - 1, 1)
        # 定义执行命令
        cmd = f"{path_vsf} -c -r -i \"{self.video_path}\" -o \"{self.temp_output_dir}\" -ces \"{self.vsf_subtitle}\" "
        cmd += f"-te {top_end} -be {bottom_end} -le {left_end} -re {right_end} -nthr {cpu_count} -nocrthr {cpu_count}"
        self.vsf_running = True
        # 计算进度
        Thread(target=count_process, daemon=True).start()
        import subprocess
        subprocess.run(cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.vsf_running = False

    def filter_watermark(self):
        """
        去除原始字幕文本中的水印区域的文本
        """
        # 获取潜在水印区域
        watermark_areas = self._detect_watermark_area()

        # 随机选择一帧, 将所水印区域标记出来，用户看图判断是否是水印区域
        cap = cv2.VideoCapture(self.video_path)
        ret, sample_frame = False, None
        for i in range(10):
            frame_no = random.randint(int(self.frame_count * 0.1), int(self.frame_count * 0.9))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, sample_frame = cap.read()
            if ret:
                break
        cap.release()

        if not ret:
            print("Error in filter_watermark: reading frame from video")
            return

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
        print(f"{interface_config['Main']['WatchPicture']}: {sample_frame_file_path}")

        area_num = ['E', 'D', 'C', 'B', 'A']
        for watermark_area in watermark_areas:
            user_input = input(f"{area_num.pop()}{str(watermark_area)} {interface_config['Main']['QuestionDelete']}").strip()
            if user_input == 'y' or user_input == '\n':
                with open(self.raw_subtitle_path, mode='r+', encoding='utf-8') as f:
                    content = f.readlines()
                    f.seek(0)
                    for i in content:
                        if i.find(str(watermark_area[0])) == -1:
                            f.write(i)
                    f.truncate()
                print(interface_config['Main']['FinishDelete'])
        print(interface_config['Main']['FinishWaterMarkFilter'])
        # 删除缓存
        if os.path.exists(sample_frame_file_path):
            os.remove(sample_frame_file_path)

    def filter_scene_text(self):
        """
        将场景里提取的文字过滤，仅保留字幕区域
        """
        # 获取潜在字幕区域
        subtitle_area = self._detect_subtitle_area()[0][0]

        # 随机选择一帧，将所水印区域标记出来，用户看图判断是否是水印区域
        cap = cv2.VideoCapture(self.video_path)
        ret, sample_frame = False, None
        for i in range(10):
            frame_no = random.randint(int(self.frame_count * 0.1), int(self.frame_count * 0.9))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, sample_frame = cap.read()
            if ret:
                break
        cap.release()

        if not ret:
            print("Error in filter_scene_text: reading frame from video")
            return

        # 为了防止有双行字幕，根据容忍度，将字幕区域y范围加高
        ymin = abs(subtitle_area[0] - config.SUBTITLE_AREA_DEVIATION_PIXEL)
        ymax = subtitle_area[1] + config.SUBTITLE_AREA_DEVIATION_PIXEL
        # 画出字幕框的区域
        cv2.rectangle(sample_frame, pt1=(0, ymin), pt2=(sample_frame.shape[1], ymax), color=(0, 0, 255), thickness=3)
        sample_frame_file_path = os.path.join(os.path.dirname(self.frame_output_dir), 'subtitle_area.jpg')
        cv2.imwrite(sample_frame_file_path, sample_frame)
        print(f"{interface_config['Main']['CheckSubArea']} {sample_frame_file_path}")

        user_input = input(f"{(ymin, ymax)} {interface_config['Main']['DeleteNoSubArea']}").strip()
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
            print(interface_config['Main']['FinishDeleteNoSubArea'])
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
        print(f"{interface_config['Main']['SubLocation']} {srt_filename}")
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
            print(f"{interface_config['Main']['SubLocation']} {srt_filename}")
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
            if config.BG_MOD == BackgroundColor.DARK:  # 深色背景
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

    def _frame_to_timecode(self, frame_no):
        """
        将视频帧转换成时间
        :param frame_no: 视频的帧号，i.e. 第几帧视频帧
        :returns: SMPTE格式时间戳 as string, 如'01:02:12:32' 或者 '01:02:12;32'
        """
        # 设置当前帧号
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, _ = cap.read()
        # 获取当前帧号对应的时间戳
        if ret:
            milliseconds = cap.get(cv2.CAP_PROP_POS_MSEC)
            if milliseconds <= 0:
                return '{0:02d}:{1:02d}:{2:02d},{3:02d}'.format(int(frame_no / (3600 * self.fps)),
                                                                int(frame_no / (60 * self.fps) % 60),
                                                                int(frame_no / self.fps % 60),
                                                                int(frame_no % self.fps))
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
        else:
            return '{0:02d}:{1:02d}:{2:02d},{3:02d}'.format(int(frame_no / (3600 * self.fps)),
                                                            int(frame_no / (60 * self.fps) % 60),
                                                            int(frame_no / self.fps % 60),
                                                            int(frame_no % self.fps))

    def _frameno_to_milliseconds(self, frame_no):
        return float(int(frame_no / self.fps * 1000))

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
        coordinates = get_coordinates(box)
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

    def _compare_ocr_result(self, result_cache, img1, img1_no, img2, img2_no):
        """
        比较两张图片预测出的字幕区域文本是否相同
        """
        if self.ocr is None:
            self.ocr = OcrRecogniser()

        # TODO: 需要大量测试, 调参diff, 实验性, 提速几倍
        # pip uninstall opencv-python
        # pip install opencv-contrib-python-headless==4.5.4.60
        # 不要装4.5.5版本! BUG!
        # hashFun = cv2.img_hash.AverageHash_create()
        # hash1 = hashFun.compute(img1)
        # hash2 = hashFun.compute(img2)
        # diff = hashFun.compare(hash1, hash2)
        # if diff <= 5:
        #     return True
        if img1_no in result_cache:
            area_text1 = result_cache[img1_no]['text']
        else:
            dt_box, rec_res = self.ocr.predict(img1)
            area_text1 = "".join(self.__get_area_text((dt_box, rec_res)))
            result_cache[img1_no] = {'text': area_text1, 'dt_box': dt_box, 'rec_res': rec_res}

        if img2_no in result_cache:
            area_text2 = result_cache[img2_no]['text']
        else:
            dt_box, rec_res = self.ocr.predict(img2)
            area_text2 = "".join(self.__get_area_text((dt_box, rec_res)))
            result_cache[img2_no] = {'text': area_text2, 'dt_box': dt_box, 'rec_res': rec_res}
        delete_no_list = []
        for no in result_cache:
            if no < min(img1_no, img2_no) - 10:
                delete_no_list.append(no)
        for no in delete_no_list:
            del result_cache[no]
        if ratio(area_text1, area_text2) > config.THRESHOLD_TEXT_SIMILARITY:
            return True
        else:
            return False

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

    def empty_cache(self):
        """
        删除字幕提取过程中所有生产的缓存文件
        """
        if os.path.exists(self.temp_output_dir):
            shutil.rmtree(self.temp_output_dir, True)


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    # 提示用户输入视频路径
    video_path = input(f"{interface_config['Main']['InputVideo']}").strip()
    # 提示用户输入字幕区域
    try:
        y_min, y_max, x_min, x_max = map(int, input(
            f"{interface_config['Main']['ChooseSubArea']} (ymin ymax xmin xmax)：").split())
        subtitle_area = (y_min, y_max, x_min, x_max)
    except ValueError as e:
        subtitle_area = None
    # 新建字幕提取对象
    se = SubtitleExtractor(video_path, subtitle_area)
    # 开始提取字幕
    se.run()
