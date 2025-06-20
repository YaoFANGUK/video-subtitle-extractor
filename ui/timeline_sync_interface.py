
import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QWidget, QFileDialog, QHBoxLayout, QVBoxLayout, QSizePolicy
from qfluentwidgets import (PushButton, ConfigValidator, ConfigItem, PlainTextEdit,
                          FluentIcon, CardWidget, SettingCardGroup, PushSettingCard, InfoBar)
from backend.config import tr
from backend.tools.python_runner import AsyncPythonRunner
from ui.icon.my_fluent_icon import MyFluentIcon

class Option:
    def __init__(self):
        self.source_video = ConfigItem("TimelineSync", "SushiSourceVideo", "", ConfigValidator())
        self.source_subtitle = ConfigItem("TimelineSync", "SushiSourceSubtitle", "", ConfigValidator())
        self.destination_video = ConfigItem("TimelineSync", "SushiDestinationVideo", "", ConfigValidator())

class TimelineSyncInterface(QWidget):

    append_log_signal = Signal(str)
    def __init__(self, parent):
        super().__init__()
        
        self.auto_scroll = True
        self.python_runner = AsyncPythonRunner()
        self.python_runner.register_callback(
            stdout = self.append_log_signal.emit,
            stderr = self.append_log_signal.emit,
            complete = self.on_complete
        )
        self.option = Option()

        self.setContentsMargins(16, 16, 16, 16)

        self.expandLayout = QVBoxLayout(self)

        self.advanced_group = SettingCardGroup(tr["TimelineSync"]["Title"], self)
        self.advanced_group.cardLayout.setSpacing(6)
        
        self.source_video = PushSettingCard(
            text=tr["TimelineSync"]["ChooseFile"],
            icon=FluentIcon.VIDEO,
            title=tr["TimelineSync"]["SourceVideoTitle"],
            content=tr["TimelineSync"]["SourceVideoDesc"] if not self.option.source_video.value else self.option.source_video.value,
            parent=self.advanced_group
        )
        self.source_video.clicked.connect(lambda: self.choose_path(self.source_video, tr["TimelineSync"]["SourceVideoTitle"], self.option.source_video, tr["TimelineSync"]["SourceVideoDesc"]))
        self.advanced_group.addSettingCard(self.source_video)

        self.source_subtitle = PushSettingCard(
            text=tr["TimelineSync"]["ChooseFile"],
            icon=FluentIcon.FONT,
            title=tr["TimelineSync"]["SourceSubtitleTitle"],
            content=tr["TimelineSync"]["SourceSubtitleDesc"] if not self.option.source_subtitle.value else self.option.source_subtitle.value,
            parent=self.advanced_group
        )
        self.source_subtitle.clicked.connect(lambda: self.choose_path(self.source_subtitle, tr["TimelineSync"]["SourceSubtitleTitle"], self.option.source_subtitle, tr["TimelineSync"]["SourceSubtitleDesc"]))
        self.advanced_group.addSettingCard(self.source_subtitle)
        
        self.destination_video = PushSettingCard(
            text=tr["TimelineSync"]["ChooseFile"],
            icon=FluentIcon.MOVIE,
            title=tr["TimelineSync"]["DestinationVideoTitle"],
            content=tr["TimelineSync"]["DestinationVideoDesc"] if not self.option.destination_video.value else self.option.destination_video.value,
            parent=self.advanced_group
        )
        self.destination_video.clicked.connect(lambda: self.choose_path(self.destination_video, tr["TimelineSync"]["DestinationVideoTitle"], self.option.destination_video, tr["TimelineSync"]["DestinationVideoDesc"]))
        self.advanced_group.addSettingCard(self.destination_video)

        self.expandLayout.addWidget(self.advanced_group)

        # 操作按钮容器
        button_container = CardWidget()
        button_container.setMinimumHeight(60)
        button_container.setStyleSheet("""
            background-color: #fcfdfe;
            border: 1px solid #eeeff0;
            border-radius: 6px;
        """)
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(16, 16, 16, 16)
        button_layout.setSpacing(8)
        button_layout.setAlignment(Qt.AlignmentFlag.AlignLeft) 
        
        self.run_button = PushButton(tr['SubtitleExtractorGUI']['Run'], self)
        self.run_button.setIcon(FluentIcon.PLAY)
        self.run_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.run_button.clicked.connect(self.run_button_clicked)
        button_layout.addWidget(self.run_button)
        
        self.stop_button = PushButton(tr['SubtitleExtractorGUI']['Stop'], self)
        self.stop_button.setIcon(MyFluentIcon.Stop)
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self.stop_button_clicked)
        button_layout.addWidget(self.stop_button)

        button_container.setLayout(button_layout)
        self.expandLayout.addWidget(button_container, 0)

         # 输出文本区域
        self.output_text = PlainTextEdit()
        self.output_text.setMinimumHeight(150)
        self.output_text.setReadOnly(True)
        self.output_text.document().setDocumentMargin(10)        
        # 连接滚动条值变化信号
        self.output_text.verticalScrollBar().valueChanged.connect(self.on_scroll_change)
        output_container = CardWidget(self)
        output_layout = QVBoxLayout()
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_text, 1)
        output_container.setLayout(output_layout)
        self.expandLayout.addWidget(output_container, 1)

        self.append_log_signal.connect(self.append_output)
        
    def choose_path(self, target, name, option, placeholder):
        """选择保存目录"""
        last_source_video = "./" if not option.value else option.value
        selected = QFileDialog.getOpenFileName(
            self, name, last_source_video)
        if len(selected) >= 1 and selected[0] == "":
            return
        path, _ = selected 
        option.value = path
        target.setContent(placeholder if not option.value else option.value)

    def on_scroll_change(self, value):
        """监控滚动条位置变化"""
        scrollbar = self.output_text.verticalScrollBar()
        # 如果滚动到底部，启用自动滚动
        if value == scrollbar.maximum():
            self.auto_scroll = True
        # 如果用户向上滚动，禁用自动滚动
        elif self.auto_scroll and value < scrollbar.maximum():
            self.auto_scroll = False
       
    def append_output(self, text):
        """添加文本到输出区域并控制滚动
        Args:
            text: 要输出的内容
        """
        text = text.strip()
        self.output_text.appendPlainText(text)
        print(text)  # 保持原始的 print 行为
        # 如果启用了自动滚动，则滚动到底部
        if self.auto_scroll:
            scrollbar = self.output_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def stop_button_clicked(self):
        self.python_runner.stop()
        self.stop_button.setVisible(False)
        self.run_button.setVisible(True)

    def run_button_clicked(self):
        """运行按钮点击事件"""
        self.python_runner.stop()

        if not self.is_selected_file_exists(tr["TimelineSync"]["SourceVideoTitle"], self.option.source_video.value):
            return
        if not self.is_selected_file_exists(tr["TimelineSync"]["SourceSubtitleTitle"], self.option.source_subtitle.value):
            return
        if not self.is_selected_file_exists(tr["TimelineSync"]["DestinationVideoTitle"], self.option.destination_video.value):
            return

        self.python_runner.start("sushi", cwd="./backend/", args=[
            '--src', self.option.source_video.value,
            '--dst', self.option.destination_video.value,
            '--script', self.option.source_subtitle.value,
            '--output', self.output_path
        ], python_args=["-u"])
        self.stop_button.setVisible(True)
        self.run_button.setVisible(False)

    def is_selected_file_exists(self, name, path):
        if not os.path.exists(path):
            InfoBar.error(
                title=tr['TimelineSync']['Error'],
                content=tr['TimelineSync']['UnableToLocateFile'].format(name, path),
                parent=self,
                duration=3000
            )
            return False
        return True

    @property
    def output_path(self):
        return os.path.join(os.path.dirname(self.option.destination_video.value), f"{Path(self.option.destination_video.value).stem}{os.path.splitext(self.option.source_subtitle.value)[-1]}")

    def on_complete(self, return_code):
        self.stop_button.setVisible(False)
        self.run_button.setVisible(True)
        if return_code == 0:
            self.append_log_signal.emit(f"Synced subtitle: {self.output_path}")
            return
