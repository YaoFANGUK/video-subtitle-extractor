from PySide6 import QtWidgets
from qfluentwidgets import (FluentWindow, PushButton, Slider, ProgressBar, PlainTextEdit,
                          setTheme, Theme, FluentIcon, CardWidget, SettingCardGroup,
                          ComboBoxSettingCard, SwitchSettingCard, RangeSettingCard,
                          PushSettingCard, PrimaryPushSettingCard, OptionsSettingCard,
                          FolderListSettingCard, HyperlinkCard, ColorSettingCard, 
                          CustomColorSettingCard)
from backend.config import config, tr, HARDWARD_ACCELERATION_OPTION

class SettingInterface(QtWidgets.QVBoxLayout):

    def __init__(self, parent):
        super().__init__()
        self.setContentsMargins(16, 16, 16, 16)
        
        # 界面语言设置
        self.interface_combo = ComboBoxSettingCard(
            configItem=config.interface,
            icon=FluentIcon.LANGUAGE,
            title=tr["LanguageModeGUI"]["InterfaceLanguage"],
            content="",
            parent=parent,
            texts=config.intefaceTexts.keys(),
        )
        self.addWidget(self.interface_combo)
        
        # 字幕语言设置
        self.language_combo = ComboBoxSettingCard(
            configItem=config.language,
            icon=FluentIcon.FONT,
            title=tr["LanguageModeGUI"]["SubtitleLanguage"],
            content="",
            parent=parent,
            texts=[tr['Language'][i] for i in config.language.validator.options],
        )
        self.addWidget(self.language_combo)

        # 识别模式设置
        self.mode_combo = ComboBoxSettingCard(
            configItem=config.mode,
            icon=FluentIcon.SETTING,
            title=tr["LanguageModeGUI"]["Mode"],
            content="",
            parent=parent,
            texts=[tr['Mode'][i] for i in config.mode.validator.options],
        )
        self.addWidget(self.mode_combo)
           
        # 是否启用硬件加速
        self.hardware_acceleration = SwitchSettingCard(
            configItem=config.hardwareAcceleration,
            icon=FluentIcon.SPEED_HIGH, 
            title=tr["Setting"]["HardwareAcceleration"],
            content=tr["Setting"]["HardwareAccelerationDesc"],
            parent=parent
        )
        self.addWidget(self.hardware_acceleration)
        # 如果硬件加速选项被禁用, 设置硬件加速为False并只读
        if not HARDWARD_ACCELERATION_OPTION:
            self.hardware_acceleration.switchButton.setChecked(False)
            self.hardware_acceleration.switchButton.setEnabled(False)
            config.set(config.hardwareAcceleration, False)
        
        # 是否生成TXT文本字幕
        self.generate_txt_switch = SwitchSettingCard(
            configItem=config.generateTxt,
            icon=FluentIcon.DOCUMENT,
            title=tr["Setting"]["GenerateTxt"],
            content="",
            parent=parent
        )
        self.addWidget(self.generate_txt_switch)
        
        # 是否重新分词
        self.word_segmentation = SwitchSettingCard(
            configItem=config.wordSegmentation,
            icon=FluentIcon.EDIT, 
            title=tr["Setting"]["WordSegmentation"],
            content=tr["Setting"]["WordSegmentationDesc"],
            parent=parent
        )
        self.addWidget(self.word_segmentation)
        
        # 添加一些空间
        self.addStretch(1)
    
    def reset_setting(self):
        """重置所有设置为默认值"""
        # 这里需要实现重置逻辑
        pass