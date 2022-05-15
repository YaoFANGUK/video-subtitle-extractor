# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao 
@Time    : 2021/3/24 9:36 上午
@FileName: config.py
@desc: 项目配置文件，可以在这里调参，牺牲时间换取精确度，或者牺牲准确度换取时间
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import configparser
import os
import re
import time
from pathlib import Path
from fsplit.filesplit import Filesplit
from paddle import fluid
from tools.constant import *

# 判断代码路径是否合法
IS_LEGAL_PATH = True
config = configparser.ConfigParser()
MODE_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.ini')
if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.ini')):
    # 如果没有配置文件，默认使用中文
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.ini'), mode='w', encoding='utf-8') as f:
        f.write('[DEFAULT]\n')
        f.write('Interface = 简体中文\n')
        f.write('Language = ch\n')
        f.write('Mode = fast')
config.read(MODE_CONFIG_PATH, encoding='utf-8')


interface_config = configparser.ConfigParser()
INTERFACE_KEY_NAME_MAP = {
    '简体中文': 'ch',
    '繁體中文': 'ch_tra',
    'English': 'en',
}
interface_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interface',
                              f"{INTERFACE_KEY_NAME_MAP[config['DEFAULT']['Interface']]}.ini")
interface_config.read(interface_file, encoding='utf-8')
# 设置识别语言
REC_CHAR_TYPE = config['DEFAULT']['Language']

# 设置识别模式
MODE_TYPE = config['DEFAULT']['Mode']
ACCURATE_MODE_ON = False
if MODE_TYPE == 'accurate':
    ACCURATE_MODE_ON = True
if MODE_TYPE == 'fast':
    ACCURATE_MODE_ON = False

# 是否使用GPU
# 使用GPU可以提速20倍+，你要是有N卡你就改成 True
USE_GPU = False
# 如果paddlepaddle编译了gpu的版本
if fluid.is_compiled_with_cuda():
    # 查看是否有可用的gpu
    if len(fluid.cuda_places()) > 0:
        # 如果有GPU则使用GPU
        USE_GPU = True

# --------------------- 请你不要改 start-----------------------------
# 项目的base目录
BASE_DIR = str(Path(os.path.abspath(__file__)).parent)
# 是否包含中文
if re.search(r"[\u4e00-\u9fa5]+", BASE_DIR):
    IS_LEGAL_PATH = False
# 是否包含空格
if re.search(r"\s", BASE_DIR):
    IS_LEGAL_PATH = False
while not IS_LEGAL_PATH:
    print(interface_config['Main']['IllegalPathWarning'])
    time.sleep(3)
# 模型文件目录
# 文本检测模型
DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
# 设置文本识别模型 + 字典
REC_MODEL_BASE = os.path.join(BASE_DIR, 'models')
# 默认字典路径为中文
DICT_BASE = os.path.join(BASE_DIR, 'ppocr', 'utils', 'dict')
REC_IMAGE_SHAPE = '3,48,320'

# 如果设置了识别文本语言类型，则设置为对应的语言
if REC_CHAR_TYPE in ('ch', 'japan', 'korean', 'en', 'EN_symbol', 'french', 'german', 'it', 'es', 'pt', 'ru', 'ar',
                     'ta', 'ug', 'fa', 'ur', 'rs_latin', 'oc', 'rs_cyrillic', 'bg', 'uk', 'be', 'te', 'kn', 'ch_tra', 'hi', 'mr', 'ne', 'EN'):
    # 定义文本检测模型
    if REC_CHAR_TYPE == 'en':
        DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, 'en_det')
    else:
        if USE_GPU:
            DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, 'ch_det')
        else:
            DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, 'ch_det_fast')
    # 定义文本识别模型
    if USE_GPU:
        REC_MODEL_PATH = os.path.join(BASE_DIR, 'models', f'{REC_CHAR_TYPE}_rec')
        if REC_CHAR_TYPE == 'ch':
            REC_IMAGE_SHAPE = '3,32,320'
    else:
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, f'{REC_CHAR_TYPE}_rec_fast')
        # 没有快速版的就使用一般版
        if not os.path.exists(REC_MODEL_PATH):
            REC_MODEL_PATH = os.path.join(BASE_DIR, 'models', f'{REC_CHAR_TYPE}_rec')
    # 定义字典路径
    DICT_PATH = os.path.join(DICT_BASE, f'{REC_CHAR_TYPE}_dict.txt')

    # 查看该路径下是否有文本模型识别完整文件，没有的话合并小文件生成完整文件
    if 'inference.pdiparams' not in (os.listdir(REC_MODEL_PATH)):
        fs = Filesplit()
        fs.merge(input_dir=REC_MODEL_PATH)

# --------------------- 请你不要改 end-----------------------------


# --------------------- 请根据自己的实际情况改 start-----------------
# 使用快速字幕检测算法时，背景颜色
BG_MOD = BackgroundColor.DARK
# 黑色背景被减矩阵阈值
BG_VALUE_DARK = 200
# 其他背景颜色被减矩阵阈值
BG_VALUE_OTHER = 63
# ROI比例
ROI_RATE = 0.4

# 默认字幕出现区域为下方
SUBTITLE_AREA = SubtitleArea.UNKNOWN

# 余弦相似度阈值
# 数值越小生成的视频帧越少，相对提取速度更快但生成的字幕越不精准
# 1表示最精准，每一帧视频帧都进行字幕检测与提取，生成的字幕最精准
# 0.925表示，当视频帧1与视频帧2相似度高达92.5%时，视频帧2将直接pass，不字检测与提取视频帧2的字幕
COSINE_SIMILARITY_THRESHOLD = 0.95 if SUBTITLE_AREA == SubtitleArea.UNKNOWN else 0.9

# 当前帧与之后的多少帧比较
FRAME_COMPARE_TIMES = 10

# 每一秒抓取多少帧进行OCR识别
EXTRACT_FREQUENCY = 3
# 每几帧抽取一帧进行OCR识别
EXTRACT_INTERVAL = 8

# 欧式距离相似值
EUCLIDEAN_SIMILARITY_THRESHOLD = 0.9

# 容忍的像素点偏差
PIXEL_TOLERANCE_Y = 50  # 允许检测框纵向偏差50个像素点
PIXEL_TOLERANCE_X = 100  # 允许检测框横向偏差100个像素点

# 字幕区域偏移量
SUBTITLE_AREA_DEVIATION_PIXEL = 50

# 最有可能出现的水印区域
WATERMARK_AREA_NUM = 5

# 文本相似度阈值
# 用于去重时判断两行字幕是不是同一行，这个值越高越严格。 e.g. 0.99表示100个字里面有99各个字一模一样才算相似
# 采用动态算法实现相似度阈值判断: 对于短文本要求较低的阈值，对于长文本要求较高的阈值
THRESHOLD_TEXT_SIMILARITY = 0.8

# 字幕提取中置信度低于0.8的不要
DROP_SCORE = 0.8

# 字幕区域允许偏差, 0为不允许越界, 0.03表示可以越界3%
SUB_AREA_DEVIATION_RATE = 0.03

# 输出丢失的字幕帧, 仅简体中文,繁体中文,日文,韩语有效, 默认将调试信息输出到: 视频路径/loss
DEBUG_OCR_LOSS = True

# --------------------- 请根据自己的实际情况改 end-----------------------------

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
