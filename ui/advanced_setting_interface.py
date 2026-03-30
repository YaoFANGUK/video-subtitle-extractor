"""
@desc: 高级设置页面
"""
import sys

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtWidgets import QFileDialog
from qfluentwidgets import (ScrollArea, ExpandLayout, CardWidget, SubtitleLabel,
                           FluentIcon, NavigationWidget, NavigationItemPosition,
                           SettingCardGroup, RangeSettingCard, SwitchSettingCard,
                           HyperlinkCard, PrimaryPushSettingCard, ComboBoxSettingCard, PushSettingCard,
                           MessageBox)
from backend.config import config, tr, VERSION, PROJECT_HOME_URL, PROJECT_ISSUES_URL, PROJECT_RELEASES_URL
from backend.tools.version_service import VersionService
from backend.tools.concurrent import TaskExecutor
from backend.tools.constant import VideoSubFinderDecoder

class AdvancedSettingInterface(ScrollArea):
    """高级设置页面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.version_manager = VersionService()
        self.__initWidget()

    def __initWidget(self):
        # 创建滚动内容的容器
        self.scrollWidget = QtWidgets.QWidget(self)
        self.expandLayout = ExpandLayout(self.scrollWidget)
        
        # 设置滚动区域属性
        self.setWidget(self.scrollWidget)
        self.enableTransparentBackground()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        
        # 设置滚动区域样式以适应主题
        self.setAttribute(QtCore.Qt.WA_StyledBackground)
        
        # 设置UI
        self.setup_ui()
        self.setup_layout()

    def setup_layout(self):
        self.advanced_group.addSettingCard(self.rec_batch_number)
        self.advanced_group.addSettingCard(self.max_batch_size)
        self.advanced_group.addSettingCard(self.subtitle_area)
        self.advanced_group.addSettingCard(self.extract_frequency)
        self.advanced_group.addSettingCard(self.tolerant_pixel_y)
        self.advanced_group.addSettingCard(self.tolerant_pixel_x)
        self.advanced_group.addSettingCard(self.subtitle_area_deviation_pixel)
        self.advanced_group.addSettingCard(self.waterark_area_num)
        self.advanced_group.addSettingCard(self.threshold_text_similarity)
        self.advanced_group.addSettingCard(self.drop_score)
        self.advanced_group.addSettingCard(self.subtitle_area_deviation_rate)
        self.advanced_group.addSettingCard(self.save_directory)
        self.advanced_group.addSettingCard(self.check_update_on_startup)
        self.expandLayout.addWidget(self.advanced_group)

        self.video_sub_finder_group.addSettingCard(self.video_sub_finder_cpu_cores)
        self.video_sub_finder_group.addSettingCard(self.video_sub_finder_decoder)
        self.expandLayout.addWidget(self.video_sub_finder_group)

        self.dev_group.addSettingCard(self.debug_ocr_loss)
        self.dev_group.addSettingCard(self.debug_no_delete_cache)
        self.dev_group.addSettingCard(self.delete_empty_time_stamp)
        self.expandLayout.addWidget(self.dev_group)

        self.about_group.addSettingCard(self.feedback)
        self.about_group.addSettingCard(self.copyright)
        self.about_group.addSettingCard(self.project_link)
        self.expandLayout.addWidget(self.about_group)
       
        self.expandLayout.setSpacing(16)
        self.expandLayout.setContentsMargins(16, 16, 16, 48)
        
    def setup_ui(self):
        """设置UI"""
        # 高级设置组
        self.advanced_group = SettingCardGroup(tr["Setting"]["AdvancedSetting"], self.scrollWidget)
        # VideoSubFinder设置组
        self.video_sub_finder_group = SettingCardGroup(tr["Setting"]["VideoSubFinderSetting"], self.scrollWidget)
        # 开发设置组  
        self.dev_group = SettingCardGroup(tr["Setting"]["DevSetting"], self.scrollWidget)
        # 关于设置组  
        self.about_group = SettingCardGroup(tr["Setting"]["AboutSetting"], self.scrollWidget)
        
        # 每张图中同时识别的文本框数量
        self.rec_batch_number = RangeSettingCard(
            configItem=config.recBatchNumber,
            icon=FluentIcon.SEARCH,
            title=tr["Setting"]["RecBatchNumber"],
            content=tr["Setting"]["RecBatchNumberDesc"],
            parent=self.advanced_group
        )
        # DB算法每个batch识别多少张
        self.max_batch_size = RangeSettingCard(
            configItem=config.maxBatchSize,
            icon=FluentIcon.SEARCH_MIRROR,
            title=tr["Setting"]["MaxBatchSize"],
            content=tr["Setting"]["MaxBatchSizeDesc"],
            parent=self.advanced_group
        )
        # 字幕出现区域
        self.subtitle_area = ComboBoxSettingCard(
            configItem=config.subtitleArea,
            icon=FluentIcon.VIEW,
            title=tr["Setting"]["SubtitleArea"],
            content=tr["Setting"]["SubtitleAreaDesc"],
            parent=self.advanced_group,
            texts=tr['SubtitleArea'].values(),
        )
        # 每一秒抓取多少帧进行OCR识别
        self.extract_frequency = RangeSettingCard(
            configItem=config.extractFrequency,
            icon=FluentIcon.SPEED_HIGH,
            title=tr["Setting"]["ExtractFrequency"],
            content=tr["Setting"]["ExtractFrequencyDesc"],
            parent=self.advanced_group
        )
        # 容忍的像素点偏差
        self.tolerant_pixel_y = RangeSettingCard(
            configItem=config.tolerantPixelY,
            icon=FluentIcon.ARROW_DOWN,
            title=tr["Setting"]["TolerantPixelY"],
            content=tr["Setting"]["TolerantPixelYDesc"],
            parent=self.advanced_group
        )
        self.tolerant_pixel_x = RangeSettingCard(
            configItem=config.tolerantPixelX,
            icon=FluentIcon.RIGHT_ARROW,
            title=tr["Setting"]["TolerantPixelX"],
            content=tr["Setting"]["TolerantPixelXDesc"],
            parent=self.advanced_group
        )
        # 字幕区域偏移量
        self.subtitle_area_deviation_pixel = RangeSettingCard(
            configItem=config.subtitleAreaDeviationPixel,
            icon=FluentIcon.MOVE,
            title=tr["Setting"]["SubtitleAreaDeviationPixel"],
            content=tr["Setting"]["SubtitleAreaDeviationPixelDesc"],
            parent=self.advanced_group
        )
        # 最有可能出现的水印区域
        self.waterark_area_num = RangeSettingCard(
            configItem=config.waterarkAreaNum,
            icon=FluentIcon.PHOTO,
            title=tr["Setting"]["WaterarkAreaNum"],
            content=tr["Setting"]["WaterarkAreaNumDesc"],
            parent=self.advanced_group
        )
        # 文本相似度阈值
        self.threshold_text_similarity = RangeSettingCard(
            configItem=config.thresholdTextSimilarity,
            icon=FluentIcon.DICTIONARY,
            title=tr["Setting"]["ThresholdTextSimilarity"],
            content=tr["Setting"]["ThresholdTextSimilarityDesc"],
            parent=self.advanced_group
        )
        # 字幕提取中置信度低于0.75的不要
        self.drop_score = RangeSettingCard(
            configItem=config.dropScore,
            icon=FluentIcon.ACCEPT_MEDIUM,
            title=tr["Setting"]["DropScore"],
            content=tr["Setting"]["DropScoreDesc"],
            parent=self.advanced_group
        )
        # 字幕区域允许偏差
        self.subtitle_area_deviation_rate = RangeSettingCard(
            configItem=config.subtitleAreaDeviationRate,
            icon=FluentIcon.PIE_SINGLE,
            title=tr["Setting"]["SubtitleAreaDeviationRate"],
            content=tr["Setting"]["SubtitleAreaDeviationRateDesc"],
            parent=self.advanced_group
        )
        # 视频保存路径
        self.save_directory = PushSettingCard(
            text=tr["Setting"]["ChooseDirectory"],
            icon=FluentIcon.DOWNLOAD,
            title=tr["Setting"]["SaveDirectory"],
            content=tr["Setting"]["SaveDirectoryDefault"] if not config.saveDirectory.value else config.saveDirectory.value,
            parent=self.advanced_group
        )
        self.save_directory.clicked.connect(self.choose_save_directory)
        # 启动时检查应用更新
        self.check_update_on_startup = SwitchSettingCard(
            configItem=config.checkUpdateOnStartup,
            icon=FluentIcon.UPDATE,
            title=tr["Setting"]["CheckUpdateOnStartup"],
            content=tr["Setting"]["CheckUpdateOnStartupDesc"],
            parent=self.advanced_group
        )
        # VideoSubFinder CPU核心数
        self.video_sub_finder_cpu_cores = RangeSettingCard(
            configItem=config.videoSubFinderCpuCores,
            icon=FluentIcon.SPEED_MEDIUM,
            title=tr["Setting"]["VideoSubFinderCpuCores"],
            content=tr["Setting"]["VideoSubFinderCpuCoresDesc"],
            parent=self.video_sub_finder_group
        )
        # VideoSubFinder 视频解码组件
        self.video_sub_finder_decoder = ComboBoxSettingCard(
            configItem=config.videoSubFinderDecoder,
            icon=FluentIcon.VIDEO,
            title=tr["Setting"]["VideoSubFinderDecoder"],
            content=tr["Setting"]["VideoSubFinderDecoderDesc"],
            texts=[item.value for item in VideoSubFinderDecoder],
            parent=self.video_sub_finder_group
        )
        # 输出丢失的字幕帧
        self.debug_ocr_loss = SwitchSettingCard(
            configItem=config.debugOcrLoss,
            icon=FluentIcon.QUESTION,
            title=tr["Setting"]["DebugOcrLoss"],
            content=tr["Setting"]["DebugOcrLossDesc"],
            parent=self.dev_group
        )
        # 是否不删除缓存数据
        self.debug_no_delete_cache = SwitchSettingCard(
            configItem=config.debugNoDeleteCache,
            icon=FluentIcon.FOLDER,
            title=tr["Setting"]["DebugNoDeleteCache"],
            content=tr["Setting"]["DebugNoDeleteCacheDesc"],
            parent=self.dev_group
        )
        # 是否删除空时间轴
        self.delete_empty_time_stamp = SwitchSettingCard(
            configItem=config.deleteEmptyTimeStamp,
            icon=FluentIcon.DELETE,
            title=tr["Setting"]["DeleteEmptyTimeStamp"],
            content=tr["Setting"]["DeleteEmptyTimeStampDesc"],
            parent=self.dev_group
        )
        # 添加反馈链接
        self.feedback = PrimaryPushSettingCard(
            text=tr["Setting"]["FeedbackButton"],
            icon=FluentIcon.MAIL,
            title=tr["Setting"]["FeedbackTitle"],
            content=tr["Setting"]["FeedbackDesc"],
            parent=self.about_group
        )
        self.feedback.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(
            QtCore.QUrl(PROJECT_ISSUES_URL)
        ))
        # 添加版权信息
        self.copyright = PrimaryPushSettingCard(
            text=tr["Setting"]["CopyrightButton"],
            icon=FluentIcon.MAIL,
            title=tr["Setting"]["CopyrightTitle"],
            content=tr["Setting"]["CopyrightDesc"].format(VERSION),
            parent=self.about_group
        )
        self.copyright.clicked.connect(lambda: self.check_update())
        # 添加项目链接
        self.project_link = HyperlinkCard(
            url=PROJECT_HOME_URL,
            text=PROJECT_HOME_URL,
            icon=FluentIcon.GITHUB,
            title=tr["Setting"]["ProjectLinkTitle"],
            content=tr["Setting"]["ProjectLinkDesc"],
            parent=self.about_group
        )

    def show_message_box(self, title: str, content: str, showYesButton=False, yesSlot=None):
        """ show message box """
        w = MessageBox(title, content, self)
        if not showYesButton:
            w.cancelButton.setText(self.tr('Close'))
            w.yesButton.hide()
            w.buttonLayout.insertStretch(0, 1)

        if w.exec() and yesSlot is not None:
            yesSlot()

    def check_update(self, ignore=False):
        """ check software update

        Parameters
        ----------
        ignore: bool
            ignore message box when no updates are available
        """
        TaskExecutor.runTask(self.version_manager.has_new_version).then(
            lambda success: self.on_version_info_fetched(success, ignore))

    def on_version_info_fetched(self, success, ignore=False):
        if success:
            self.show_message_box(
                tr["Setting"]["UpdatesAvailableTitle"],
                tr["Setting"]["UpdatesAvailableDesc"].format(self.version_manager.lastest_version),
                True,
                lambda: QtGui.QDesktopServices.openUrl(
                    QtCore.QUrl(PROJECT_RELEASES_URL)
                )
            )
        elif not ignore:
            self.show_message_box(
                tr["Setting"]["NoUpdatesAvailableTitle"],
                tr["Setting"]["NoUpdatesAvailableDesc"],
            )
    
    def choose_save_directory(self):
        """选择保存目录"""
        last_save_directory = "./" if not config.saveDirectory.value else config.saveDirectory.value
        folder = QFileDialog.getExistingDirectory(
            self, tr['Setting']['ChooseDirectory'], last_save_directory)
        if not folder:
            folder = ""

        config.set(config.saveDirectory, folder)
        self.save_directory.setContent(tr["Setting"]["SaveDirectoryDefault"] if not config.saveDirectory.value else config.saveDirectory.value)
            
    def macos_scrollarea_issue_workaround(self):
        if sys.platform != "darwin":
            return
        self.verticalScrollBar().setValue(0)
        self.scrollWidget.adjustSize()
        self.expandLayout.update()
        self.expandLayout.activate()
        
    def resizeEvent(self, event):
        # macos_scrollarea_issue_workaround
        if sys.platform == "darwin":
            self.verticalScrollBar().setValue(0)
        super().resizeEvent(event)