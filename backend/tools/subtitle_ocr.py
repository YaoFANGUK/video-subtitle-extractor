import os
import re
from multiprocessing import Queue, Process
import cv2
from PIL import ImageFont, ImageDraw, Image
from tqdm import tqdm
from tools.ocr import OcrRecogniser, get_coordinates
from tools.constant import SubtitleArea
from tools import constant
from threading import Thread
import queue
from shapely.geometry import Polygon
from types import SimpleNamespace
import shutil
import numpy as np
from collections import namedtuple


def extract_subtitles(data, text_recogniser, img, raw_subtitle_file,
                      sub_area, options, dt_box_arg, rec_res_arg, ocr_loss_debug_path):
    """
    提取视频帧中的字幕信息
    """
    # 从参数中获取检测框与检测结果
    dt_box = dt_box_arg
    rec_res = rec_res_arg
    # 如果没有检测结果，则获取检测结果
    if dt_box is None or rec_res is None:
        dt_box, rec_res = text_recogniser.predict(img)
        # rec_res格式为： ("hello", 0.997)
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
            # 初始化超界偏差为0
            overflow_area_rate = 0
            # 用户指定的字幕区域
            sub_area_polygon = sub_area_to_polygon(sub_area)
            # 识别出的字幕区域
            coordinate_polygon = coordinate_to_polygon(coordinate)
            # 计算两个区域是否有交集交集
            intersection = sub_area_polygon.intersection(coordinate_polygon)
            # 如果有交集
            if not intersection.is_empty:
                # 计算越界允许偏差
                overflow_area_rate = ((sub_area_polygon.area + coordinate_polygon.area - intersection.area) / sub_area_polygon.area) - 1
                # 如果越界比例低于设定阈值且该行文本识别的置信度高于设定阈值
                if overflow_area_rate <= options.SUB_AREA_DEVIATION_RATE and prob > options.DROP_SCORE:
                    # 保留该帧
                    selected = True
                    line += f'{str(data["i"]).zfill(8)}\t{coordinate}\t{text}\n'
                    raw_subtitle_file.write(f'{str(data["i"]).zfill(8)}\t{coordinate}\t{text}\n')
            # 保存丢掉的识别结果
            loss_info = namedtuple('loss_info', 'text prob overflow_area_rate coordinate selected')
            loss_list.append(loss_info(text, prob, overflow_area_rate, coordinate, selected))
        else:
            raw_subtitle_file.write(f'{str(data["i"]).zfill(8)}\t{coordinate}\t{text}\n')
    # 输出调试信息
    dump_debug_info(options, line, img, loss_list, ocr_loss_debug_path, sub_area, data)


def dump_debug_info(options, line, img, loss_list, ocr_loss_debug_path, sub_area, data):
    loss = False
    if options.DEBUG_OCR_LOSS and options.REC_CHAR_TYPE in ('ch', 'japan ', 'korea', 'ch_tra'):
        loss = len(line) > 0 and re.search(r'[\u4e00-\u9fa5\u3400-\u4db5\u3130-\u318F\uAC00-\uD7A3\u0800-\u4e00]', line) is None
    if loss:
        if not os.path.exists(ocr_loss_debug_path):
            os.makedirs(ocr_loss_debug_path, mode=0o777, exist_ok=True)
        img = cv2.rectangle(img, (sub_area[2], sub_area[0]), (sub_area[3], sub_area[1]), constant.BGR_COLOR_BLUE, 2)
        for loss_info in loss_list:
            coordinate = loss_info.coordinate
            color = constant.BGR_COLOR_GREEN if loss_info.selected else constant.BGR_COLOR_RED
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
FONT = ImageFont.truetype(FONT_PATH, 20)


def paint_chinese_opencv(im, chinese, pos, color):
    img_pil = Image.fromarray(im)
    fill_color = color  # (color[2], color[1], color[0])
    position = pos
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, chinese, font=FONT, fill=fill_color)
    img = np.asarray(img_pil)
    return img


def ocr_task_consumer(ocr_queue, raw_subtitle_path, sub_area, video_path, options):
    """
    消费者： 消费ocr_queue，将ocr队列中的数据取出，进行ocr识别，写入字幕文件中
    :param ocr_queue (current_frame_no当前帧帧号, frame 视频帧, dt_box检测框, rec_res识别结果)
    :param raw_subtitle_path
    :param sub_area
    :param video_path
    :param options
    """
    data = {'i': 1}
    # 初始化文本识别对象
    text_recogniser = OcrRecogniser()
    # 丢失字幕的存储路径
    ocr_loss_debug_path = os.path.join(os.path.abspath(os.path.splitext(video_path)[0]), 'loss')
    # 删除之前的缓存垃圾
    if os.path.exists(ocr_loss_debug_path):
        shutil.rmtree(ocr_loss_debug_path, True)

    with open(raw_subtitle_path, mode='w+', encoding='utf-8') as raw_subtitle_file:
        while True:
            try:
                frame_no, frame, dt_box, rec_res = ocr_queue.get(block=True)
                if frame_no == -1:
                    return
                data['i'] = frame_no
                extract_subtitles(data, text_recogniser, frame, raw_subtitle_file, sub_area, options, dt_box,
                                  rec_res, ocr_loss_debug_path)
            except Exception as e:
                print(e)
                break


def ocr_task_producer(ocr_queue, task_queue, progress_queue, video_path, raw_subtitle_path):
    """
    生产者：负责生产用于OCR识别的数据，将需要进行ocr识别的数据加入ocr_queue中
    :param ocr_queue (current_frame_no当前帧帧号, frame 视频帧, dt_box检测框, rec_res识别结果)
    :param task_queue (total_frame_count总帧数, current_frame_no当前帧帧号, dt_box检测框, rec_res识别结果, subtitle_area字幕区域)
    :param progress_queue
    :param video_path
    :param raw_subtitle_path
    """
    cap = cv2.VideoCapture(video_path)
    tbar = None
    while True:
        try:
            # 从任务队列中提取任务信息
            total_frame_count, current_frame_no, dt_box, rec_res, total_ms, default_subtitle_area = task_queue.get(block=True)
            progress_queue.put(current_frame_no)
            if tbar is None:
                tbar = tqdm(total=round(total_frame_count), position=1)
            # current_frame 等于-1说明所有视频帧已经读完
            if current_frame_no == -1:
                # ocr识别队列加入结束标志
                ocr_queue.put((-1, None, None, None))
                # 更新进度条
                tbar.update(tbar.total - tbar.n)
                break
            tbar.update(round(current_frame_no - tbar.n))
            # 设置当前视频帧
            # 如果total_ms不为空，则使用了VSF提取字幕
            if total_ms is not None:
                cap.set(cv2.CAP_PROP_POS_MSEC, total_ms)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_no)
            # 读取视频帧
            ret, frame = cap.read()
            # 如果读取成功
            if ret:
                # 根据默认字幕位置，则对视频帧进行裁剪，裁剪后处理
                if default_subtitle_area is not None:
                    frame = frame_preprocess(default_subtitle_area, frame)
                ocr_queue.put((current_frame_no, frame, dt_box, rec_res))
        except Exception as e:
            print(e)
            break
    cap.release()


def subtitle_extract_handler(task_queue, progress_queue, video_path, raw_subtitle_path, sub_area, options):
    """
    创建并开启一个视频帧提取线程与一个ocr识别线程
    :param task_queue 任务队列，(total_frame_count总帧数, current_frame_no当前帧, dt_box检测框, rec_res识别结果, subtitle_area字幕区域)
    :param progress_queue 进度队列
    :param video_path 视频路径
    :param raw_subtitle_path 原始字幕文件路径
    :param sub_area 字幕区域
    :param options 选项
    """
    # 删除缓存
    if os.path.exists(raw_subtitle_path):
        os.remove(raw_subtitle_path)
    # 创建一个OCR队列，大小建议值8-20
    ocr_queue = queue.Queue(20)
    # 创建一个OCR事件生产者线程
    ocr_event_producer_thread = Thread(target=ocr_task_producer,
                                       args=(ocr_queue, task_queue, progress_queue, video_path, raw_subtitle_path,),
                                       daemon=True)
    # 创建一个OCR事件消费者提取线程
    ocr_event_consumer_thread = Thread(target=ocr_task_consumer,
                                       args=(ocr_queue, raw_subtitle_path, sub_area, video_path, options,),
                                       daemon=True)
    # 开启消费者线程
    ocr_event_producer_thread.start()
    # 开启生产者线程
    ocr_event_consumer_thread.start()
    # join方法让主线程任务结束之后，进入阻塞状态，一直等待其他的子线程执行结束之后，主线程再终止
    ocr_event_producer_thread.join()
    ocr_event_consumer_thread.join()



def async_start(video_path, raw_subtitle_path, sub_area, options):
    """
    开始进程处理异步任务
    options.REC_CHAR_TYPE
    options.DROP_SCORE
    options.SUB_AREA_DEVIATION_RATE
    options.DEBUG_OCR_LOSS
    """
    assert 'REC_CHAR_TYPE' in options, "options缺少参数：REC_CHAR_TYPE"
    assert 'DROP_SCORE' in options, "options缺少参数: DROP_SCORE'"
    assert 'SUB_AREA_DEVIATION_RATE' in options, "options缺少参数: SUB_AREA_DEVIATION_RATE"
    assert 'DEBUG_OCR_LOSS' in options, "options缺少参数: DEBUG_OCR_LOSS"
    # 创建一个任务队列
    # 任务格式为：(total_frame_count总帧数, current_frame_no当前帧, dt_box检测框, rec_res识别结果, subtitle_area字幕区域)
    task_queue = Queue()
    # 创建一个进度更新队列
    progress_queue = Queue()
    # 新建一个进程
    p = Process(target=subtitle_extract_handler,
                args=(task_queue, progress_queue, video_path, raw_subtitle_path, sub_area, SimpleNamespace(**options),))
    # 启动进程
    p.start()
    return p, task_queue, progress_queue


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
