import os
import sys
from enum import Enum

from qfluentwidgets import getIconColor, Theme, FluentIconBase


def _get_base_path():
    if hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MyFluentIcon(FluentIconBase, Enum):
    Stop = "stop"

    def path(self, theme=Theme.AUTO):
        # getIconColor() return "white" or "black" according to current theme
        return os.path.join(_get_base_path(), 'ui', 'icon', f'{self.value}_{getIconColor(theme)}.svg')
