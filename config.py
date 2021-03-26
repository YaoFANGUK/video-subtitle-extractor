# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao 
@Time    : 2021/3/24 9:36 上午
@FileName: config.py
@desc: 项目配置文件，可以在这里调参，牺牲时间换取精确度，或者牺牲准确度换取时间
"""
import os
from pathlib import Path
from enum import Enum
from fsplit.filesplit import Filesplit
from paddle import fluid
fluid.install_check.run_check()


# --------------------- 请你不要改 start-----------------------------
# 项目的base目录
BASE_DIR = str(Path(os.path.abspath(__file__)).parent)

# 模型文件目录
# 文本检测模型
DET_MODEL_PATH = os.path.join(BASE_DIR, 'backend', 'models', 'ch_det')
# 文本识别模型
REC_MODEL_PATH = os.path.join(BASE_DIR, 'backend', 'models', 'ch_rec')

# 查看该路径下是否有文本模型识别完整文件，没有的话合并小文件生成完整文件
if 'inference.pdiparams' not in (os.listdir(REC_MODEL_PATH)):
    fs = Filesplit()
    fs.merge(input_dir=REC_MODEL_PATH)

# 字典路径
DICT_PATH = os.path.join(BASE_DIR, 'backend', 'ppocr', 'utils', 'ppocr_keys_v1.txt')


# 默认字幕出现的大致区域
class SubtitleArea(Enum):
    # 字幕区域出现在下半部分
    LOWER_PART = 0
    # 字幕区域出现在上半部分
    UPPER_PART = 1
    # 不知道字幕区域可能出现的位置
    UNKNOWN = 2
    # 明确知道字幕区域出现的位置
    CUSTOM = 3
# --------------------- 请你不要改 end-----------------------------


# --------------------- 请根据自己的实际情况改 start-----------------------------
# 是否使用GPU
# 使用GPU可以提速20倍+，你要是有N卡你就改成 True
USE_GPU = False
if len(fluid.cuda_places()) > 0:
    # 如果有GPU则使用GPU
    USE_GPU = True
    print('使用GPU进行加速')

# 默认字幕出现区域为下方
SUBTITLE_AREA = SubtitleArea.LOWER_PART

# 余弦相似度阈值
# 数值越小生成的视频帧越少，相对提取速度更快但生成的字幕越不精准
# 1表示最精准，每一帧视频帧都进行字幕检测与提取，生成的字幕最精准
# 0.925表示，当视频帧1与视频帧2相似度高达92.5%时，视频帧2将直接pass，不字检测与提取视频帧2的字幕
COSINE_SIMILARITY_THRESHOLD = 0.95 if SUBTITLE_AREA == SubtitleArea.UNKNOWN else 0.9

# 当前帧与之后的多少帧比较
FRAME_COMPARE_TIMES = 10

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
# 用于去重时判断两行字幕是不是统一行
TEXT_SIMILARITY_THRESHOLD = 0.95
# --------------------- 请根据自己的实际情况改 end-----------------------------

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
