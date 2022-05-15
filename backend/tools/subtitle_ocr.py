import os
import re
from multiprocessing import Queue, Process
import cv2
from PIL import ImageFont, ImageDraw, Image
from tqdm import tqdm
from tools.ocr import OcrRecogniser, get_coordinates
from tools.constant import SubtitleArea
from threading import Thread
import queue
from shapely.geometry import Polygon
from types import SimpleNamespace
import shutil
import numpy as np
from collections import namedtuple

BGR_COLOR_GREEN = (0, 0xff, 0)
BGR_COLOR_BLUE = (0xff, 0, 0)
BGR_COLOR_RED = (0, 0, 0xff)
BGR_COLOR_WHITE = (0xff, 0xff, 0xff)


def extract_subtitles(data, text_recogniser, img, raw_subtitle_file,
                      sub_area, options, dt_box, rec_res, ocr_loss_debug_path):
    """
        提取视频帧中的字幕信息
    """
    # 获取检测结果
    if dt_box is None or rec_res is None:
        dt_box, rec_res = text_recogniser.predict(img)
    else:
        pass
    # 获取文本坐标
    coordinates = get_coordinates(dt_box)
    # 将结果写入txt文本中
    if options.REC_CHAR_TYPE == 'en':
        # 如果识别语言为英文，则去除中文
        text_res = [(re.sub('[\u4e00-\u9fa5]', '', res[0]), res[1]) for res in rec_res]
    else:
        text_res = [(res[0], res[1]) for res in rec_res]
    line = ''
    loss_list = []
    for content, coordinate in zip(text_res, coordinates):
        text = content[0]
        prob = content[1]
        if sub_area is not None:
            selected = False
            overflow_area_rate = 0
            sub_area_polygon = sub_area_to_polygon(sub_area)
            coordinate_polygon = coordinate_to_polygon(coordinate)
            # 计算交集
            intersection = sub_area_polygon.intersection(coordinate_polygon)
            # 如果有交集
            if not intersection.is_empty:
                # 计算越界允许偏差
                overflow_area_rate = ((sub_area_polygon.area + coordinate_polygon.area - intersection.area)\
                                     / sub_area_polygon.area) - 1
                if overflow_area_rate <= options.SUB_AREA_DEVIATION_RATE:
                    if prob > options.DROP_SCORE:
                        selected = True
                        line += f'{str(data["i"]).zfill(8)}\t{coordinate}\t{text}\n'
                        raw_subtitle_file.write(f'{str(data["i"]).zfill(8)}\t{coordinate}\t{text}\n')
            loss_info = namedtuple('loss_info', 'text prob overflow_area_rate coordinate selected')
            loss_list.append(loss_info(text, prob, overflow_area_rate, coordinate, selected))
        else:
            raw_subtitle_file.write(f'{str(data["i"]).zfill(8)}\t{coordinate}\t{text}\n')
    # 输出调试信息
    dump_debug_info(options, line, img, loss_list, ocr_loss_debug_path, sub_area, data)
    data["i"] += 1


def dump_debug_info(options, line, img, loss_list, ocr_loss_debug_path, sub_area, data):
    loss = False
    if options.DEBUG_OCR_LOSS and options.REC_CHAR_TYPE in ('ch', 'japan ', 'korea', 'ch_tra'):
        loss = len(line) > 0 and re.search(r'[\u4e00-\u9fa5\u3400-\u4db5\u3130-\u318F\uAC00-\uD7A3\u0800-\u4e00]', line) is None
    if loss:
        if not os.path.exists(ocr_loss_debug_path):
            os.makedirs(ocr_loss_debug_path, mode=0o777, exist_ok=True)
        img = cv2.rectangle(img, (sub_area[2], sub_area[0]), (sub_area[3], sub_area[1]), BGR_COLOR_BLUE, 2)
        for loss_info in loss_list:
            coordinate = loss_info.coordinate
            color = BGR_COLOR_GREEN if loss_info.selected else BGR_COLOR_RED
            text = f"[{loss_info.text}] prob:{loss_info.prob:.4f} or:{loss_info.overflow_area_rate:.2f}"
            img = paint_chinese_opencv(img, text, pos=(coordinate[0], coordinate[2] - 30), color=color)
            img = cv2.rectangle(img, (coordinate[0], coordinate[2]), (coordinate[1], coordinate[3]), color, 2)
        cv2.imwrite(os.path.join(os.path.abspath(ocr_loss_debug_path), f'{str(data["i"]).zfill(8)}.png'), img)


def sub_area_to_polygon(sub_area):
    s_ymin = sub_area[0]
    s_ymax = sub_area[1]
    s_xmin = sub_area[2]
    s_xmax = sub_area[3]
    return Polygon([[s_xmin, s_ymin], [s_xmax, s_ymin], [s_xmax, s_ymax], [s_xmin, s_ymax]])


def coordinate_to_polygon(coordinate):
    xmin = coordinate[0]
    xmax = coordinate[1]
    ymin = coordinate[2]
    ymax = coordinate[3]
    return Polygon([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])


FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NotoSansCJK-Bold.otf')
FONT = ImageFont.truetype(FONT_PATH,20)


def paint_chinese_opencv(im,chinese,pos,color):
    img_PIL = Image.fromarray(im)
    fillColor = color  # (color[2], color[1], color[0])
    position = pos
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, chinese, font=FONT, fill=fillColor)
    img = np.asarray(img_PIL)
    return img


def recv_video_frame_handle(ocr_queue, raw_subtitle_path, sub_area, video_path, options):
    data = {'i': 1}
    # 初始化文本识别对象
    text_recogniser = OcrRecogniser()
    ocr_loss_debug_path = os.path.join(os.path.abspath(os.path.splitext(video_path)[0]), 'loss')
    if os.path.exists(ocr_loss_debug_path):
        shutil.rmtree(ocr_loss_debug_path, True)
    with open(raw_subtitle_path, mode='w+', encoding='utf-8') as raw_subtitle_file:
        while True:
            try:
                total_ms, frame_no, frame, dt_box, rec_res = ocr_queue.get(block=True)
                if total_ms == -1:
                    return
                if frame_no > 0:
                    data['i'] = frame_no
                extract_subtitles(data, text_recogniser, frame, raw_subtitle_file,
                                  sub_area, options, dt_box, rec_res, ocr_loss_debug_path)
            except Exception as e:
                print(e)
                break


def recv_ocr_event_handle(ocr_queue, event_queue, progress_queue, video_path, raw_subtitle_path):
    # 删除缓存
    if os.path.exists(raw_subtitle_path):
        os.remove(raw_subtitle_path)
    cap = cv2.VideoCapture(video_path)
    tbar = None
    while True:
        try:
            total_ms, duration_ms, frame_no, dt_box, rec_res, subtitle_area = event_queue.get(block=True)
            progress_queue.put(total_ms)
            if tbar is None:
                tbar = tqdm(total=round(duration_ms), position=1)
            if total_ms == -1:
                ocr_queue.put((-1, None, None, None, None))
                tbar.update(tbar.total - tbar.n)
                break
            tbar.update(round(total_ms - tbar.n))
            cap.set(cv2.CAP_PROP_POS_MSEC, total_ms)
            ret, frame = cap.read()
            if ret:
                if subtitle_area is not None:
                    frame = frame_preprocess(subtitle_area, frame)
                ocr_queue.put((total_ms, frame_no, frame, dt_box, rec_res))
        except Exception as e:
            print(e)
            break
    cap.release()


def handle(event_queue, progress_queue, video_path, raw_subtitle_path, sub_area, options):
    # 建议值8-20
    ocr_queue = queue.Queue(20)
    recv_video_frame_thread = Thread(target=recv_video_frame_handle,
                                     args=(ocr_queue, raw_subtitle_path, sub_area, video_path, options,),
                                     daemon=True)
    recv_ocr_event_thread = Thread(target=recv_ocr_event_handle,
                                   args=(ocr_queue, event_queue, progress_queue, video_path, raw_subtitle_path,),
                                   daemon=True)
    recv_ocr_event_thread.start()
    recv_video_frame_thread.start()
    recv_video_frame_thread.join()
    recv_ocr_event_thread.join()


"""
    options.REC_CHAR_TYPE
    options.DROP_SCORE
    options.SUB_AREA_DEVIATION_RATE
    options.DEBUG_OCR_LOSS
"""
def async_start(video_path, raw_subtitle_path, sub_area, options):
    assert 'REC_CHAR_TYPE' in options
    assert 'DROP_SCORE' in options
    assert 'SUB_AREA_DEVIATION_RATE' in options
    assert 'DEBUG_OCR_LOSS' in options
    event_queue = Queue()
    progress_queue = Queue()
    t = Process(target=handle,
                args=(event_queue, progress_queue, video_path, raw_subtitle_path, sub_area, SimpleNamespace(**options),))
    t.start()
    return t, event_queue, progress_queue


def frame_preprocess(subtitle_area, frame):
    """
    将视频帧进行裁剪
    """
    # 对于分辨率大于1920*1080的视频，将其视频帧进行等比缩放至1280*720进行识别
    # paddlepaddle会将图像压缩为640*640
    # if self.frame_width > 1280:
    #     scale_rate = round(float(1280 / self.frame_width), 2)
    #     frames = cv2.resize(frames, None, fx=scale_rate, fy=scale_rate, interpolation=cv2.INTER_AREA)
    # 如果字幕出现的区域在下部分
    if subtitle_area == SubtitleArea.LOWER_PART:
        cropped = int(frame.shape[0] // 2)
        # 将视频帧切割为下半部分
        frame = frame[cropped:]
    # 如果字幕出现的区域在上半部分
    elif subtitle_area == SubtitleArea.UPPER_PART:
        cropped = int(frame.shape[0] // 2)
        # 将视频帧切割为下半部分
        frame = frame[:cropped]
    return frame


if __name__ == "__main__":
    pass
