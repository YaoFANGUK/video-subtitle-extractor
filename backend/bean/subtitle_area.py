
from typing import Union
from dataclasses import dataclass
from shapely.geometry import Polygon

@dataclass
class SubtitleArea:
    """
    字幕区域
    """
    ymin: Union[int, float]
    ymax: Union[int, float]
    xmin: Union[int, float]
    xmax: Union[int, float]
    # 字幕区域在视频中的位置
    ab_section: range = None
    
    def __init__(self, ymin: Union[int, float], ymax: Union[int, float], 
                 xmin: Union[int, float], xmax: Union[int, float], 
                 ab_section: range = None):
        self.ymin = ymin
        self.ymax = ymax    
        self.xmin = xmin
        self.xmax = xmax
        self.ab_section = ab_section

    def normalized(self):
        if self.xmin > self.xmax:
            self.xmin, self.xmax = self.xmax, self.xmin
        if self.ymin > self.ymax:
            self.ymin, self.ymax = self.ymax, self.ymin

    def is_empty(self):
        return self.xmin == 0 and self.xmax == 0 and self.ymin == 0 and self.ymax == 0

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin

    def in_ab_section(self, frame_idx):
        return True

    def to_polygon(self):
        return Polygon([[self.xmin, self.ymin], [self.xmax, self.ymin], [self.xmax, self.ymax], [self.xmin, self.ymax]])