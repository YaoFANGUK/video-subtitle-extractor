from enum import Enum


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


class BackgroundColor(Enum):
    # 字幕背景
    WHITE = 0
    DARK = 1
    UNKNOWN = 2


BGR_COLOR_GREEN = (0, 0xff, 0)
BGR_COLOR_BLUE = (0xff, 0, 0)
BGR_COLOR_RED = (0, 0, 0xff)
BGR_COLOR_WHITE = (0xff, 0xff, 0xff)
