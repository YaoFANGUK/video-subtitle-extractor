import os
import re
from multiprocessing import Queue, Process
import cv2
from tqdm import tqdm
from tools.ocr import OcrRecogniser, get_coordinates
from tools.constant import SubtitleArea
from threading import Thread
import queue


def extract_subtitles(data, text_recogniser, img, raw_subtitle_file, sub_area,
                      rec_char_type, drop_score, dt_box, rec_res):
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
    if rec_char_type == 'en':
        # 如果识别语言为英文，则去除中文
        text_res = [(re.sub('[\u4e00-\u9fa5]', '', res[0]), res[1]) for res in rec_res]
    else:
        text_res = [(res[0], res[1]) for res in rec_res]
    for content, coordinate in zip(text_res, coordinates):
        if sub_area is not None:
            s_ymin = sub_area[0]
            s_ymax = sub_area[1]
            s_xmin = sub_area[2]
            s_xmax = sub_area[3]
            xmin = coordinate[0]
            xmax = coordinate[1]
            ymin = coordinate[2]
            ymax = coordinate[3]
            if s_xmin <= xmin and xmax <= s_xmax and s_ymin <= ymin and ymax <= s_ymax:
                if content[1] > drop_score:
                    raw_subtitle_file.write(f'{str(data["i"]).zfill(8)}\t{coordinate}\t{content[0]}\n')
        else:
            raw_subtitle_file.write(f'{str(data["i"]).zfill(8)}\t{coordinate}\t{content[0]}\n')
    data["i"] += 1


def recv_video_frame_handle(ocr_queue, raw_subtitle_path, sub_area, rec_char_type, drop_score):
    data = {'i': 1}
    # 初始化文本识别对象
    text_recogniser = OcrRecogniser()
    with open(raw_subtitle_path, mode='w+', encoding='utf-8') as raw_subtitle_file:
        while True:
            try:
                total_ms, frame_no, frame, dt_box, rec_res = ocr_queue.get(block=True)
                if total_ms == -1:
                    return
                if frame_no > 0:
                    data['i'] = frame_no
                extract_subtitles(data, text_recogniser, frame, raw_subtitle_file, sub_area,
                                  rec_char_type, drop_score, dt_box, rec_res)

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


def handle(event_queue, progress_queue, video_path, raw_subtitle_path, sub_area, rec_char_type, drop_score):
    # 建议值8-20
    ocr_queue = queue.Queue(20)
    recv_video_frame_thread = Thread(target=recv_video_frame_handle,
                                     args=(ocr_queue, raw_subtitle_path, sub_area, rec_char_type, drop_score,),
                                     daemon=True)
    recv_ocr_event_thread = Thread(target=recv_ocr_event_handle,
                                   args=(ocr_queue, event_queue, progress_queue, video_path, raw_subtitle_path,),
                                   daemon=True)
    recv_ocr_event_thread.start()
    recv_video_frame_thread.start()
    recv_video_frame_thread.join()
    recv_ocr_event_thread.join()


def async_start(video_path, raw_subtitle_path, sub_area, rec_char_type, drop_score):
    event_queue = Queue()
    progress_queue = Queue()
    t = Process(target=handle,
                args=(event_queue, progress_queue, video_path, raw_subtitle_path, sub_area, rec_char_type, drop_score,))
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
