# -*- coding: utf-8 -*-
"""
@Author  : Fang Yao 
@Time    : 2021/3/26 2:07 下午
@FileName: gui.py
@desc: 图形化界面版的字幕提取器
"""
import tkinter as tk
from main import SubtitleExtractor
import config
from tkinter.filedialog import askopenfilename
from tkinter import ttk
import time


class SubtitleExtractorGUI:
    """
    带有图形化界面的字幕提取器
    """

    def __init__(self):
        """
        初始化窗口以及组件
        """
        # 创建主窗口用于容纳其它组件
        self.window = tk.Tk()
        # 给窗口的可视化起名字
        self.window.title('视频硬字幕提取器')
        # 获取屏幕分辨率
        self.screenwidth = self.window.winfo_screenwidth()
        self.screenheight = self.window.winfo_screenheight()
        # 设定窗口的大小(长 * 宽)
        self.window.geometry(f'{self.screenheight}x{self.screenwidth}')  # 这里的乘是小x
        # ------------------ 定义组件 ------------------------
        # 新建一个菜单栏
        self.menubar = tk.Menu(self.window)

        # 新建一个"文件"菜单项，（默认不下拉，下拉内容包括"打开", "关闭"等功能项）
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        # 将上面定义的空菜单命名为文件，放在菜单栏中，就是装入那个容器中
        self.menubar.add_cascade(label='文件', menu=self.file_menu)
        # 在"文件"菜单中加入"打开"，"关闭"等小菜单，即我们平时看到的下拉菜单，每一个小菜单对应命令操作。
        self.file_menu.add_command(label='打开视频', command=None)  # TODO: command加入回调函数
        self.file_menu.add_separator()  # 添加一条分隔线
        self.file_menu.add_command(label='关闭窗口', command=self.window.quit)

        # 新建一个"设置"菜单项，（默认不下拉，下拉内容包括"打开", "关闭"等功能项）
        self.running_menu = tk.Menu(self.menubar, tearoff=0)
        # 将setting_menu放入menubar
        self.menubar.add_cascade(label='运行', menu=self.running_menu)
        # 在"运行"菜单中加入"提取关键帧"，"提取字幕"，
        # "检测水印区域"， "过滤水印区域文本"，
        # "检测字幕区域"， "过滤非字幕区域文本"，
        # "生成字幕文件", "清除缓存文件" 等选项
        self.running_menu.add_command(label='提取关键帧', command=None)  # TODO: command加入回调函数
        self.running_menu.add_command(label='提取字幕', command=None)  # TODO: command加入回调函数
        self.running_menu.add_command(label='检测水印区域', command=None)  # TODO: command加入回调函数
        self.running_menu.add_command(label='过滤水印区域文本', command=None)  # TODO: command加入回调函数
        self.running_menu.add_command(label='检测字幕区域', command=None)  # TODO: command加入回调函数
        self.running_menu.add_command(label='过滤非字幕区域文本', command=None)  # TODO: command加入回调函数
        self.running_menu.add_command(label='生成字幕文件', command=None)  # TODO: command加入回调函数
        self.file_menu.add_separator()  # 添加一条分隔线
        # 创建二级菜单
        self.clean_cache_menu = tk.Menu(self.running_menu)
        self.running_menu.add_cascade(label='清除缓存文件', menu=self.clean_cache_menu)
        # 创建三级菜单
        self.clean_cache_menu.add_command(label='清除所有缓存文件', command=None)  # TODO: command加入回调函数
        self.clean_cache_menu.add_command(label='清除视频帧', command=None)  # TODO: command加入回调函数
        self.clean_cache_menu.add_command(label='清除raw txt文件', command=None)  # TODO: command加入回调函数

        # 将菜单添加到window中
        self.window.config(menu=self.menubar)

        # 新建一个按钮，点击后选择视频路径
        self.open_video_btn = tk.Button(self.window, text='打开视频', font=('Arial', 12), width=10, height=1,
                                        command=None)  # TODO 添加回调
        # 新建一个用于显示视频帧的画布
        self.frame_canvas = tk.Canvas(self.window, bg='white',
                                      height=self.screenheight // 3,
                                      width=self.screenwidth)
        # 新建一个用于显示输出信息的画布
        self.output_canvas = tk.Canvas(self.window, bg='white',
                                       height=self.screenheight // 3,
                                       width=self.screenwidth)
        # 新建一个进度条组件
        self.progressbar = ttk.Progressbar(self.window, length=300,
                                           orient=tk.HORIZONTAL,
                                           style='grey.Horizontal.TProgressbar')
        # 新建一个按钮组件
        self.start_btn = tk.Button(self.window, text='开始', command=self.__increment_process)

    def arrange_component(self):
        # 放置视频帧画布
        self.frame_canvas.pack()
        # 放置输入信息画布
        self.output_canvas.pack()
        # 放置按钮
        self.start_btn.pack(side='left')
        # 放置进度条组件
        self.progressbar.pack(side='right')

    def __increment_process(self):
        while True:
            if self.progressbar["value"] == 100:
                self.progressbar["value"] = 0
            for i in range(100):
                self.progressbar["value"] = i + 1
                self.window.update()
                time.sleep(0.1)

    def __switch_btn_status(self):
        """更新按钮状态"""
        pass


if __name__ == '__main__':
    # 初始化窗口对象
    window = SubtitleExtractorGUI()
    # 对组件进行布局
    window.arrange_component()

    # 主窗口循环显示
    tk.mainloop()
    # 注意，loop因为是循环的意思，window.mainloop就会让window不断的刷新，
    # 如果没有mainloop,就是一个静态的window,传入进去的值就不会有循环，
    # mainloop就相当于一个很大的while循环，有个while，每点击一次就会更新一次，所以我们必须要有循环
    # 所有的窗口文件都必须有类似的mainloop函数，mainloop是窗口文件的关键的关键。
