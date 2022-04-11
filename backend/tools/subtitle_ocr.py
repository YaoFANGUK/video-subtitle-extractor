import os
import re
from multiprocessing import Queue, Process
import cv2
from tqdm import tqdm

from tools.ocr import OcrRecogniser, get_coordinates
from tools.constant import SubtitleArea

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

def handle(queue, video_path, raw_subtitle_path, sub_area, rec_char_type, drop_score):
    # 删除缓存
    if os.path.exists(raw_subtitle_path):
        os.remove(raw_subtitle_path)
    cap = cv2.VideoCapture(video_path)
    tbar = None
    tqdm_offset = 0
    last_total_ms = 0
    # 初始化文本识别对象
    text_recogniser = OcrRecogniser()
    data = {'i': 1}
    with open(raw_subtitle_path, mode='w+', encoding='utf-8') as raw_subtitle_file:
        while True:
            try:
                total_ms, duration_ms, frame_no, dt_box, rec_res, subtitle_area = queue.get(block=True)
                if total_ms == -1:
                    tbar.update(max(0, int(duration_ms) - tqdm_offset))
                    break
                if tbar is None:
                    tbar = tqdm(total=int(duration_ms), unit='f', position=1)
                total_ms_incr = int(total_ms - last_total_ms)
                tqdm_offset += total_ms_incr
                tbar.update(total_ms_incr)
                last_total_ms = total_ms
                cap.set(cv2.CAP_PROP_POS_MSEC, total_ms)
                ret, frame = cap.read()
                if ret:
                    if subtitle_area is not None:
                        frame = frame_preprocess(subtitle_area, frame)
                    if frame_no > 0:
                        data['i'] = frame_no
                    extract_subtitles(data, text_recogniser, frame, raw_subtitle_file, sub_area,
                                      rec_char_type, drop_score, dt_box, rec_res)
            except Exception as e:
                print(e)
                break
    cap.release()


def async_start(video_path, raw_subtitle_path, sub_area, rec_char_type, drop_score):
    queue = Queue()
    t = Process(target=handle, args=(queue, video_path, raw_subtitle_path, sub_area, rec_char_type, drop_score,))
    t.start()
    return t, queue


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
