# coding=utf-8
import os, sys
import shutil
import sys
import time
import shutil  
import re

import cv2
import numpy as np
import tensorflow as tf
import codecs
from collections import Counter

import matplotlib.pyplot as plt
import glob
from PIL import Image
from cnocr import CnOcr
from fuzzywuzzy import fuzz

sys.path.append(os.getcwd())

from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector

tf.app.flags.DEFINE_string('data_path', 'data\\frames\\', '')
tf.app.flags.DEFINE_string('output_path', 'data\\text_position\\', '')
tf.app.flags.DEFINE_string('ocr_path', 'data\\to_ocr\\', '')
tf.app.flags.DEFINE_string('srt_path', 'data\\to_srt\\', '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_mlt\\', '')
FLAGS = tf.app.flags.FLAGS


def video_to_frames(path):
    videoCap = cv2.VideoCapture(path)
    # 帧频
    fps = videoCap.get(cv2.CAP_PROP_FPS)
    # 视频总帧数
    total_framesX = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 图像尺寸
    image_size = (int(videoCap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(videoCap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    print('fps: ', fps)
    print('Total Frames: ', total_framesX)
    print('Video Resolution: ', image_size)
    if total_framesX < 1:
        print('fail to read video')
        return
    else:
        print('Extracting frames, please wait...')
    # 获取文件夹目录
    ex_folder = FLAGS.data_path
    if not os.path.exists(ex_folder):
        os.mkdir(ex_folder)

    # start from the first frame
    current_frame = 1

    # 扫描字幕次数 (每秒扫描一次)
    loop_times = int(total_framesX/fps)
    # 逐帧扫描
    # loop_times = int(total_framesX)

    for i in range(loop_times):
        sucess, frame = videoCap.read()
        if frame is None:
            # print('video: %s finish at %d frame.' % (video_filename, current_frame))
            break
        im = frame[:, :, 0]
        img = Image.fromarray(im)
        timeline = str(current_frame) + ".png"
        imgname = os.path.join(ex_folder, timeline)
        img.save(imgname)
        for j in range(int(fps)):  #跳过剩下的帧，因为字幕持续时间往往1s以上
            sucess, frame = videoCap.read()
            current_frame += 1
    return fps

def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

def detect_waterprint(raw_srt_path):
    if not os.path.exists(raw_srt_path):
        print('Raw Srt File Do Not Exist')
        return
    
    f = codecs.open(raw_srt_path, mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8’编码读取
    line = f.readline()   # 以行的形式进行读取文件
    ymin = []
    ymax = []
    xmin = []
    xmax = []
    while line:
        text_position = line.split('\t')[1].split('[')[1].split(']')[0].split(', ')
        ymin.append(int(text_position[0]))
        ymax.append(int(text_position[1]))
        xmin.append(int(text_position[2]))
        xmax.append(int(text_position[3]))
        line = f.readline()
    f.close()
    waterprint_area = (Counter(ymin).most_common()[0][0], Counter(ymax).most_common()[0][0], Counter(xmin).most_common()[0][0], Counter(xmax).most_common()[0][0]) 
    return waterprint_area

def delete_waterprint(raw_srt_path, waterprint_area):
    print('We detected that the watermark area is about: '+ str(waterprint_area))
    choice = input('If the watermark area is located correctly,input "y" else Press ENTER: ')
    if choice == 'y':
        # 根据视频分辨率可以自定义偏差
        y_deviation = 100
        x_deviation = 100
        y_min_bottom = waterprint_area[0] - y_deviation
        y_min_upper = waterprint_area[0] + y_deviation
        y_max_bottom = waterprint_area[1] - y_deviation
        y_max_upper = waterprint_area[1] + y_deviation    

        x_min_bottom = waterprint_area[2] - x_deviation
        x_min_upper = waterprint_area[2] + x_deviation
        x_max_bottom = waterprint_area[3] - x_deviation
        x_max_upper = waterprint_area[3] + x_deviation   

        with open(raw_srt_path,'r',encoding='utf-8') as r:
            lines=r.readlines()
        with open(raw_srt_path,'w',encoding='utf-8') as w:
            for l in lines:
                pos = l.split('\t')[1].split('[')[1].split(']')[0].split(', ')
                y_min = int(pos[0])
                y_max = int(pos[1])
                x_min = int(pos[2])
                x_max = int(pos[3])
                count = 0
                if (y_min >= y_min_bottom) and (y_min <= y_min_upper):
                    count = count + 1
                if (y_max >= y_max_bottom) and (y_max <= y_max_upper):
                    count = count + 1
                if (x_min >= x_min_bottom) and (x_min <= x_min_upper):
                    count = count + 1
                if (x_max >= x_max_bottom) and (x_max <= x_max_upper):
                    count = count + 1
                if count < 4:
                    w.write(l) 
        print('ALL water print text are removed')
    else:
        return

def detect_subtitle_area(raw_srt_path):
    if not os.path.exists(raw_srt_path):
        print('Raw Srt File Do Not Exist')
        return
    
    f = codecs.open(raw_srt_path, mode='r', encoding='utf-8')  # 打开txt文件，以‘utf-8’编码读取
    line = f.readline()   # 以行的形式进行读取文件
    ymin = []
    ymax = []
    while line:
        text_position = line.split('\t')[1].split('[')[1].split(']')[0].split(', ')
        ymin.append(int(text_position[0]))
        ymax.append(int(text_position[1]))
        line = f.readline()
    f.close()
    subtitle_area = (Counter(ymin).most_common()[0][0], Counter(ymax).most_common()[0][0]) 
    print(subtitle_area)
    return subtitle_area

def nonsubtitle_filter(raw_srt_path, subtitle_area):
    print('We detected that the subtitle area is about: '+ str(subtitle_area))
    choice = input('If the subtitle area is located correctly,input "y" else Press ENTER: ')
    if choice == 'y':
        # 根据视频分辨率可以自定义偏差
        y_deviation = 50
        y_min_bottom = subtitle_area[0] - y_deviation
        y_min_upper = subtitle_area[0] + y_deviation
        y_max_bottom = subtitle_area[1] - y_deviation
        y_max_upper = subtitle_area[1] + y_deviation    

        with open(raw_srt_path,'r',encoding='utf-8') as r:
            lines=r.readlines()
        with open(raw_srt_path,'w',encoding='utf-8') as w:
            for l in lines:
                pos = l.split('\t')[1].split('[')[1].split(']')[0].split(', ')
                y_min = int(pos[0])
                y_max = int(pos[1])
                count = 0
                if (y_min >= y_min_bottom) and (y_min <= y_min_upper):
                    count = count + 1
                if (y_max >= y_max_bottom) and (y_max <= y_max_upper):
                    count = count + 1
                if count >= 2:
                    w.write(l) 
        print('ALL non subtitle area text are removed')
    else:
        return

def clear_buff():
    if os.path.exists(FLAGS.data_path):
        shutil.rmtree(FLAGS.data_path)
    if os.path.exists(FLAGS.output_path):
        shutil.rmtree(FLAGS.output_path)
    if os.path.exists(FLAGS.ocr_path):
        shutil.rmtree(FLAGS.ocr_path)
    if os.path.exists(FLAGS.srt_path):
        shutil.rmtree(FLAGS.srt_path)


# 文本区域范围
def text_range(txt):
    ranges = []
    text_postion_info = open('{}'.format(txt)).read().split('\n')[ : -1]
    area_num = len(text_postion_info)
    for i in range(area_num):
        text_postion_info[i] = text_postion_info[i].split(',')
        x1 = int(text_postion_info[i][0])
        y1 = int(text_postion_info[i][1])
        x2 = int(text_postion_info[i][2])
        y2 = int(text_postion_info[i][3])
        x3 = int(text_postion_info[i][4])
        y3 = int(text_postion_info[i][5])
        x4 = int(text_postion_info[i][6])
        y4 = int(text_postion_info[i][7])
        ymin = min(y1, y3)
        ymax = max(y1, y3)
        xmin = min(x1, x3)
        xmax = max(x1, x3)
        ranges.append([ymin,ymax,xmin,xmax])
    return ranges


def to_textImg():
    # 创建输入管道
    images = glob.glob(FLAGS.data_path + "*.png")
    print(images)
    txts = glob.glob(FLAGS.output_path + "*.txt")
    print(txts)
    # 排序以便对应
    images.sort(key = lambda x:x.split('\\')[-1].split('.png')[0] )
    txts.sort(key = lambda x:x.split('\\')[-1].split('.txt')[0] )
    # 输出目录
    output_folder = FLAGS.ocr_path
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for i in range(len(images)):
        img, (rh, rw) =  resize_image(cv2.imread(images[i]))
        tr = text_range(txts[i])
        if tr == []:
            continue
        for j in range(len(tr)):
            text_image = img[tr[j][0]:tr[j][1], tr[j][2]:tr[j][3]]
            output = output_folder + images[i].split('\\')[-1].split('.png')[0]
            if not os.path.exists(output):
                os.mkdir(output)
            imgname =  os.path.join(output, str(tr[j]) + '.png')
            cv2.imwrite(imgname, text_image)   



def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])

# 去除非中英文字符
def cleantxt(raw):
    fil = re.compile(u'[^0-9a-zA-Z\u4e00-\u9fa5.，,？\-“”]+', re.UNICODE)
    return fil.sub('', raw)

def to_raw_srt(path, srt_dir):
    ocr = CnOcr()
    if not os.path.exists(srt_dir):
        os.mkdir(srt_dir)
    dir_list = [int(r) for r in os.listdir(path) if r.isdigit()]
    dir_list.sort()
    file = srt_dir + '\\to_srt' + '.txt'
    if os.path.exists(file):
        os.remove(file)
    for i in dir_list:
        child_dir = path + '\\' + str(i)
        file_list = os.listdir(child_dir)
        for j in file_list:
            frame_no = str(child_dir.split('\\')[-1])
            text_position = j.split('.')[0]
            # OCR识别调用
            content = cleantxt("".join(ocr.ocr_for_single_line(child_dir + '\\' + j)))
            with open(file, 'a+',encoding='utf-8') as f:
                f.write(frame_no + '\t' + text_position + '\t' + content + '\n')
    f.close()

def text_detect():
    if os.path.exists(FLAGS.output_path):
        shutil.rmtree(FLAGS.output_path)
    os.makedirs(FLAGS.output_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    with tf.get_default_graph().as_default():
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            im_fn_list = get_images()
            for im_fn in im_fn_list:
                print('===============')
                print(im_fn)
                start = time.time()
                try:
                    im = cv2.imread(im_fn)[:, :, ::-1]
                except:
                    print("Error reading image {}!".format(im_fn))
                    continue

                img, (rh, rw) = resize_image(im)
                h, w, c = img.shape
                im_info = np.array([h, w, c]).reshape([1, 3])
                bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                       feed_dict={input_image: [img],
                                                                  input_im_info: im_info})

                textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
                scores = textsegs[:, 0]
                textsegs = textsegs[:, 1:5]

                textdetector = TextDetector(DETECT_MODE='H')
                boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
                boxes = np.array(boxes, dtype=np.int)

                cost_time = (time.time() - start)
                print("cost time: {:.2f}s".format(cost_time))

                for i, box in enumerate(boxes):
                    cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                                  thickness=2)
                img = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(FLAGS.output_path, os.path.basename(im_fn)), img[:, :, ::-1])

                with open(os.path.join(FLAGS.output_path, os.path.splitext(os.path.basename(im_fn))[0]) + ".txt",
                          "w") as f:
                    for i, box in enumerate(boxes):
                        line = ",".join(str(box[k]) for k in range(8))
                        line += "," + str(scores[i]) + "\n"
                        f.writelines(line)

def frames_to_timecode(framerate,frames):
    """
    视频 通过视频帧转换成时间
    :param framerate: 视频帧率
    :param frames: 当前视频帧数
    :return:时间（00:00:01:01）
    """
    return '{0:02d}:{1:02d}:{2:02d},{3:02d}'.format(int(frames / (3600 * framerate)),
                                                    int(frames / (60 * framerate) % 60),
                                                    int(frames / framerate % 60),
                                                    int(frames % framerate))


def generate_srtfile(raw_srt_path, fps):
    if not os.path.exists(raw_srt_path):
        print('Raw Srt File Do Not Exist')
        return
    
    #自定义重复文本判断的精度
    similarity_threshold = 92
    

    with open(raw_srt_path,'r',encoding='utf-8') as r:
        lines = r.readlines()
        
    list = [] 
    list.append(lines[0])
        
    for line in lines[1:]:
        current_frame = line.split('\t')[0]
        current_position = line.split('\t')[1]
        current_content = line.split('\t')[2]
        similarity = fuzz.ratio(current_content, list[-1].split('\t')[2])
        if similarity < similarity_threshold:
            # 不存在就添加进去
            # 此处可以添加纠错网络
            list.append(list[-1].split('\t')[0].split('-')[-1] + '-' + current_frame + '\t' + current_position + '\t' + current_content)
        else:
            # 存在则修改帧数范围
            # 此处可以添加纠错网络
            list[-1] = list[-1].split('\t')[0] + '-' + current_frame + '\t' + current_position + '\t' + current_content
            
#     f = open(raw_srt_path, 'w')
#     for i in list:
#         f.write(i)
#     f.close()
    print('Duplicates are all removed')
    
    f = open(raw_srt_path.split('.txt')[0] + '.srt', 'w',encoding='utf-8')
    line_num = 1
    for i in list:
        time_start = frames_to_timecode(fps, int(i.split('\t')[0].split('-')[0]))
        time_end = frames_to_timecode(fps, int(i.split('\t')[0].split('-')[-1]))
        content = i.split('\t')[2]
        f.write(str(line_num) + '\n' + time_start + ' --> ' + time_end + '\n' + content + '\n')
        line_num = line_num + 1
    f.close()

    print('Srt File Was Generated at: ',raw_srt_path.split('.txt')[0])


def main(argv=None):
    clear_buff()
    videopath = input("please input your video file path name:").strip()
    fps = video_to_frames(videopath)
    text_detect()
    to_textImg()
    to_raw_srt(FLAGS.ocr_path, FLAGS.srt_path)
    to_srt_path = FLAGS.srt_path + '\\to_srt.txt'
    delete_waterprint(to_srt_path, detect_waterprint(to_srt_path))
    nonsubtitle_filter(to_srt_path, detect_subtitle_area(to_srt_path))
    generate_srtfile(to_srt_path, fps)

if __name__ == '__main__':
    tf.app.run()
