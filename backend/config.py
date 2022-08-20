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
        f.write('Mode = fast')
settings_config.read(MODE_CONFIG_PATH, encoding='utf-8')

# 读取interface下的语言配置,e.g. ch.ini
interface_config = configparser.ConfigParser()
INTERFACE_KEY_NAME_MAP = {
    '简体中文': 'ch',
    '繁體中文': 'chinese_cht',
    'English': 'en',
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
# 模型文件目录
# 默认模型版本 V3
MODEL_VERSION = 'V3'
# 文本检测模型
DET_MODEL_BASE = os.path.join(BASE_DIR, 'models')
# 设置文本识别模型 + 字典
REC_MODEL_BASE = os.path.join(BASE_DIR, 'models')
# 默认字典路径为中文
DICT_BASE = os.path.join(BASE_DIR, 'ppocr', 'utils', 'dict')
# V3模型默认图形识别的shape为3, 48, 320
REC_IMAGE_SHAPE = '3,48,320'
REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec')
DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_det')

# 如果设置了识别文本语言类型，则设置为对应的语言
if REC_CHAR_TYPE in ('ch', 'japan', 'korean', 'en', 'chinese_cht', 'EN_symbol', 'french', 'german', 'it', 'es', 'pt',
                     'cyrillic', 'ru', 'ar', 'ta', 'ug', 'fa', 'ur', 'rs_latin', 'oc', 'bg', 'uk', 'be', 'te', 'kn',
                     'hi', 'mr', 'ne', 'EN'):
    # 不管有无GPU和是否开启精准模式，默认使用大模型
    # 定义文本识别模型
    REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec')
    # 如果当前模型版本没有大模型，则切换为V2版本
    if not os.path.exists(REC_MODEL_PATH):
        MODEL_VERSION = 'V2'
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec')
    # 如果V2版本也没有大模型版本，则切换V3的fast版本
    if not os.path.exists(REC_MODEL_PATH):
        MODEL_VERSION = 'V3'
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec_fast')
    # 如果没有V3的fast版本，则切换V2的fast版本
    if not os.path.exists(REC_MODEL_PATH):
        MODEL_VERSION = 'V2'
        REC_MODEL_PATH = os.path.join(REC_MODEL_BASE, MODEL_VERSION, f'{REC_CHAR_TYPE}_rec_fast')
    # 定义文本检测模型
    tmp_dir = REC_MODEL_PATH.replace(os.path.dirname(REC_MODEL_PATH), '').split('_')
    if len(tmp_dir) > 3:
        tmp_dir = tmp_dir[-3:]
    tmp_dir[1] = 'det'
    DET_MODEL_PATH = os.path.join(os.path.dirname(REC_MODEL_PATH), "_".join(tmp_dir))
    if not os.path.exists(DET_MODEL_PATH):
        tmp_dir[0] = 'ch'
        DET_MODEL_PATH = os.path.join(os.path.dirname(REC_MODEL_PATH), "_".join(tmp_dir))
    if MODEL_VERSION == 'V2':
        DET_MODEL_PATH = os.path.join(DET_MODEL_BASE, MODEL_VERSION, 'ch_det')
    # 定义字典路径
    DICT_PATH = os.path.join(DICT_BASE, f'{REC_CHAR_TYPE}_dict.txt')
    # 定义图像识别shape
    if MODEL_VERSION == 'V2':
        REC_IMAGE_SHAPE = '3,32,320'
    else:
        REC_IMAGE_SHAPE = '3,48,320'

    # 查看该路径下是否有文本模型识别完整文件，没有的话合并小文件生成完整文件
    if 'inference.pdiparams' not in (os.listdir(REC_MODEL_PATH)):
        fs = Filesplit()
        fs.merge(input_dir=REC_MODEL_PATH)
# ×××××××××××××××××××× [不要改]读取语言、模型路径、字典路径 end ××××××××××××××××××××


# --------------------- 请根据自己的实际情况改 start-----------------
# 使用快速字幕检测算法(_analyse_subtitle_frame)时，背景颜色
BG_MOD = BackgroundColor.DARK
# 黑色背景被减矩阵阈值
BG_VALUE_DARK = 200
# 其他背景颜色被减矩阵阈值
BG_VALUE_OTHER = 63
# ROI比例
ROI_RATE = 0.4

# 默认字幕出现区域为下方
DEFAULT_SUBTITLE_AREA = SubtitleArea.UNKNOWN

# 余弦相似度阈值
# 数值越小生成的视频帧越少，相对提取速度更快但生成的字幕越不精准
# 1表示最精准，每一帧视频帧都进行字幕检测与提取，生成的字幕最精准
# 0.925表示，当视频帧1与视频帧2相似度高达92.5%时，视频帧2将直接pass，不检测与提取视频帧2的字幕
COSINE_SIMILARITY_THRESHOLD = 0.95 if DEFAULT_SUBTITLE_AREA == SubtitleArea.UNKNOWN else 0.9

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

# --------------------- 请根据自己的实际情况改 end-----------------------------

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
