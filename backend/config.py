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


# 项目的base目录
BASE_DIR = str(Path(os.path.abspath(__file__)).parent)

# ×××××××××××××××××××× [不要改]读取配置文件 start ××××××××××××××××××××
# 读取settings.ini配置
settings_config = configparser.ConfigParser()
MODE_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.ini')
if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.ini')):
    # 如果没有配置文件，默认使用中文
    with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'settings.ini'), mode='w', encoding='utf-8') as f:
        f.write('[DEFAULT]\n')
        f.write('Interface = 简体中文\n')
        f.write('Language = ch\n')
        f.write('Mode = fast\n')
        f.write('IgnoreVertical = false')
settings_config.read(MODE_CONFIG_PATH, encoding='utf-8')

# 读取interface下的语言配置,e.g. ch.ini
interface_config = configparser.ConfigParser()
INTERFACE_KEY_NAME_MAP = {
    '简体中文': 'ch',
    '繁體中文': 'chinese_cht',
    'English': 'en',
    '한국어': 'ko',
    '日本語': 'japan',
    'Tiếng Việt': 'vi',
    'Español': 'es'
}
interface_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interface',
                              f"{INTERFACE_KEY_NAME_MAP[settings_config['DEFAULT']['Interface']]}.ini")
interface_config.read(interface_file, encoding='utf-8')
# ×××××××××××××××××××× [不要改]读取配置文件 end ××××××××××××××××××××


# ×××××××××××××××××××× [不要改]判断程序运行路径是否合法 start ××××××××××××××××××××
# 程序运行路径如果包含中文或者空格，运行过程在程序可能会存在bug，因此需要检查路径合法性
# 默认为合法路径
IS_LEGAL_PATH = True
# 如果路径包含中文，设置路径为非法
if re.search(r"[\u4e00-\u9fa5]+", BASE_DIR):
    IS_LEGAL_PATH = False
# 如果路径包含空格，设置路径为非法
if re.search(r"\s", BASE_DIR):
    IS_LEGAL_PATH = False
# 如果为程序存放在非法路径则一直提示用户路径不合法
while not IS_LEGAL_PATH:
    print(interface_config['Main']['IllegalPathWarning'])
    time.sleep(3)
# ×××××××××××××××××××× [不要改]判断程序运行路径是否合法 end ××××××××××××××××××××


# ×××××××××××××××××××× [不要改]判断是否使用GPU start ××××××××××××××××××××
# 是否使用GPU
USE_GPU = False
# 如果paddlepaddle编译了gpu的版本
if fluid.is_compiled_with_cuda():
    # 查看是否有可用的gpu
    if len(fluid.cuda_places()) > 0:
        # 如果有GPU则使用GPU
        USE_GPU = True
# ×××××××××××××××××××× [不要改]判断是否使用GPU start ××××××××××××××××××××


# ×××××××××××××××××××× [不要改]读取语言、模型路径、字典路径 start ××××××××××××××××××××
# 设置识别语言
REC_CHAR_TYPE = settings_config['DEFAULT']['Language']

# 设置识别模式
MODE_TYPE = settings_config['DEFAULT']['Mode']
ACCURATE_MODE_ON = False
if MODE_TYPE == 'accurate':
    ACCURATE_MODE_ON = True
if MODE_TYPE == 'fast':
    ACCURATE_MODE_ON = False
if MODE_TYPE == 'auto':
    if USE_GPU:
        ACCURATE_MODE_ON = True
    else:
        ACCURATE_MODE_ON = False
# 模型文件目录
# 默认模型版本 V4
MODEL_VERSION = 'V4'
# 文本检测模型
DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
# 设置文本识别模型 + 字典
REC_MODEL_BASE = os.path.join(BASE_DIR, 'models')
# 默认字典路径为中文
DICT_BASE = os.path.join(BASE_DIR, 'ppocr', 'utils', 'dict')
# V3, V4模型默认图形识别的shape为3, 48, 320
REC_IMAGE_SHAPE = '3,48,320'
REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec')
DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_det')

LATIN_LANG = [
    'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
    'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
    'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
    'sw', 'tl', 'tr', 'uz', 'vi', 'latin', 'german', 'french'
]
ARABIC_LANG = ['ar', 'fa', 'ug', 'ur']
CYRILLIC_LANG = [
    'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
    'dar', 'inh', 'che', 'lbe', 'lez', 'tab', 'cyrillic'
]
DEVANAGARI_LANG = [
    'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
    'sa', 'bgc', 'devanagari'
]
OTHER_LANG = [
    'ch', 'japan', 'korean', 'en', 'ta', 'kn', 'te', 'ka',
    'chinese_cht',
]
MULTI_LANG = LATIN_LANG + ARABIC_LANG + CYRILLIC_LANG + DEVANAGARI_LANG + \
             OTHER_LANG

# 定义字典路径
DICT_PATH = os.path.join(DICT_BASE, f'{REC_CHAR_TYPE}_dict.txt')
DET_MODEL_FAST_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det_fast')


# 如果设置了识别文本语言类型，则设置为对应的语言
if REC_CHAR_TYPE in MULTI_LANG:
    # 定义文本检测与识别模型
    # 使用快速模式时，调用轻量级模型
    if MODE_TYPE == 'fast':
        DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det_fast')
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec_fast')
    # 使用自动模式时，检测有没有使用GPU，根据GPU判断模型
    elif MODE_TYPE == 'auto':
        # 如果使用GPU，则使用大模型
        if USE_GPU:
            DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')
            # 英文模式的ch模型识别效果好于fast
            if REC_CHAR_TYPE == 'en':
                REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'ch_rec')
                DICT_PATH = os.path.join(DICT_BASE, f'ch_dict.txt')
            else:
                REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec')
        else:
            DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det_fast')
            REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec_fast')
    else:
        DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec')
    # 如果默认版本(V4)没有大模型，则切换为默认版本(V4)的fast模型
    if not os.path.exists(REC_MODEL_PATH):
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec_fast')
    # 如果默认版本(V4)既没有大模型，又没有fast模型，则使用V3版本的大模型
    if not os.path.exists(REC_MODEL_PATH):
        MODEL_VERSION = 'V3'
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec')
    # 如果V3版本没有大模型，则使用V3版本的fast模型
    if not os.path.exists(REC_MODEL_PATH):
        MODEL_VERSION = 'V3'
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec_fast')

    if REC_CHAR_TYPE in LATIN_LANG:
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'latin_rec_fast')
        DICT_PATH = os.path.join(DICT_BASE, f'latin_dict.txt')
    elif REC_CHAR_TYPE in ARABIC_LANG:
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'arabic_rec_fast')
        DICT_PATH = os.path.join(DICT_BASE, f'arabic_dict.txt')
    elif REC_CHAR_TYPE in CYRILLIC_LANG:
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'cyrillic_rec_fast')
        DICT_PATH = os.path.join(DICT_BASE, f'cyrillic_dict.txt')
    elif REC_CHAR_TYPE in DEVANAGARI_LANG:
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'devanagari_rec_fast')
        DICT_PATH = os.path.join(DICT_BASE, f'devanagari_dict.txt')

    # 定义图像识别shape
    if MODEL_VERSION == 'V2':
        REC_IMAGE_SHAPE = '3,32,320'
    else:
        REC_IMAGE_SHAPE = '3,48,320'

    # 查看该路径下是否有文本模型识别完整文件，没有的话合并小文件生成完整文件
    if 'inference.pdiparams' not in (os.listdir(REC_MODEL_PATH)):
        fs = Filesplit()
        fs.merge(input_dir=REC_MODEL_PATH)
    # 查看该路径下是否有文本模型识别完整文件，没有的话合并小文件生成完整文件
    if 'inference.pdiparams' not in (os.listdir(DET_MODEL_PATH)):
        fs = Filesplit()
        fs.merge(input_dir=DET_MODEL_PATH)
# ×××××××××××××××××××× [不要改]读取语言、模型路径、字典路径 end ××××××××××××××××××××


# --------------------- 请根据自己的实际情况改 start-----------------
# 每张图中同时识别6个文本框中的文本，GPU显存越大，该数值可以设置越大
REC_BATCH_NUM = 6
# DB算法每个batch识别多少张，默认为10
MAX_BATCH_SIZE = 10

# 默认字幕出现区域为下方
DEFAULT_SUBTITLE_AREA = SubtitleArea.UNKNOWN

# 每一秒抓取多少帧进行OCR识别
EXTRACT_FREQUENCY = 3

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
# 如：文本较短，人民、入民，0.5就算相似
THRESHOLD_TEXT_SIMILARITY = 0.8

# 字幕提取中置信度低于0.75的不要
DROP_SCORE = 0.75

# 字幕区域允许偏差, 0为不允许越界, 0.03表示可以越界3%
SUB_AREA_DEVIATION_RATE = 0

# 输出丢失的字幕帧, 仅简体中文,繁体中文,日文,韩语有效, 默认将调试信息输出到: 视频路径/loss
DEBUG_OCR_LOSS = False

# 是否不删除缓存数据，以方便调试
DEBUG_NO_DELETE_CACHE = False

# 是否删除空时间轴
DELETE_EMPTY_TIMESTAMP = True

# 是否重新分词, 用于解决没有语句没有空格
WORD_SEGMENTATION = True

# 是否忽略竖排字幕
IGNORE_VERTICAL_LINE = False
if 'IgnoreVertical' in settings_config['DEFAULT'].keys() and 
    settings_config['DEFAULT']['IgnoreVertical'] == 'true':
    IGNORE_VERTICAL_LINE = True
    IGNORE_VERTICAL_TEXT_LEN_THRESHOLD = 2

# --------------------- 请根据自己的实际情况改 end-----------------------------

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
