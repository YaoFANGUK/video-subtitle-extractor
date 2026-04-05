# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import copy_metadata

datas = [('C:\\Users\\yao\\Projects\\video-subtitle-extractor\\backend\\interface', 'backend/interface'), ('C:\\Users\\yao\\Projects\\video-subtitle-extractor\\backend\\configs', 'backend/configs'), ('C:\\Users\\yao\\Projects\\video-subtitle-extractor\\backend\\models', 'backend/models'), ('C:\\Users\\yao\\Projects\\video-subtitle-extractor\\backend\\subfinder', 'backend/subfinder'), ('C:\\Users\\yao\\Projects\\video-subtitle-extractor\\design', 'design'), ('C:\\Users\\yao\\Projects\\video-subtitle-extractor\\ui\\icon', 'ui/icon'), ('C:\\Users\\yao\\Projects\\video-subtitle-extractor\\backend\\tools\\NotoSansCJK-Bold.otf', 'backend/tools')]
binaries = []
datas += collect_data_files('paddleocr')
datas += collect_data_files('paddlex')
datas += collect_data_files('paddle')
datas += collect_data_files('qfluentwidgets')
datas += collect_data_files('PySide6')
datas += collect_data_files('wordsegment')
datas += copy_metadata('aistudio-sdk')
datas += copy_metadata('altgraph')
datas += copy_metadata('annotated-doc')
datas += copy_metadata('annotated-types')
datas += copy_metadata('anyio')
datas += copy_metadata('bce-python-sdk')
datas += copy_metadata('certifi')
datas += copy_metadata('chardet')
datas += copy_metadata('charset-normalizer')
datas += copy_metadata('click')
datas += copy_metadata('colorama')
datas += copy_metadata('colorlog')
datas += copy_metadata('crc32c')
datas += copy_metadata('darkdetect')
datas += copy_metadata('filelock')
datas += copy_metadata('fsspec')
datas += copy_metadata('future')
datas += copy_metadata('h11')
datas += copy_metadata('hf-xet')
datas += copy_metadata('httpcore')
datas += copy_metadata('httpx')
datas += copy_metadata('huggingface_hub')
datas += copy_metadata('idna')
datas += copy_metadata('ImageIO')
datas += copy_metadata('imageio-ffmpeg')
datas += copy_metadata('imagesize')
datas += copy_metadata('je-showinfilemanager')
datas += copy_metadata('lazy-loader')
datas += copy_metadata('Levenshtein')
datas += copy_metadata('lmdb')
datas += copy_metadata('markdown-it-py')
datas += copy_metadata('mdurl')
datas += copy_metadata('modelscope')
datas += copy_metadata('networkx')
datas += copy_metadata('numpy')
datas += copy_metadata('opencv-contrib-python')
datas += copy_metadata('opencv-python')
datas += copy_metadata('opt-einsum')
datas += copy_metadata('packaging')
datas += copy_metadata('paddleocr')
datas += copy_metadata('paddlepaddle')
datas += copy_metadata('paddlex')
datas += copy_metadata('pandas')
datas += copy_metadata('pefile')
datas += copy_metadata('pillow')
datas += copy_metadata('pip')
datas += copy_metadata('prettytable')
datas += copy_metadata('protobuf')
datas += copy_metadata('psutil')
datas += copy_metadata('pyclipper')
datas += copy_metadata('pycryptodome')
datas += copy_metadata('pydantic')
datas += copy_metadata('pydantic_core')
datas += copy_metadata('Pygments')
datas += copy_metadata('pyinstaller')
datas += copy_metadata('pyinstaller-hooks-contrib')
datas += copy_metadata('pypdfium2')
datas += copy_metadata('PySide6')
datas += copy_metadata('PySide6_Addons')
datas += copy_metadata('PySide6_Essentials')
datas += copy_metadata('PySide6-Fluent-Widgets')
datas += copy_metadata('PySideSix-Frameless-Window')
datas += copy_metadata('pysrt')
datas += copy_metadata('python-bidi')
datas += copy_metadata('python-dateutil')
datas += copy_metadata('pywin32')
datas += copy_metadata('pywin32-ctypes')
datas += copy_metadata('PyYAML')
datas += copy_metadata('py-cpuinfo')
datas += copy_metadata('RapidFuzz')
datas += copy_metadata('requests')
datas += copy_metadata('rich')
datas += copy_metadata('ruamel.yaml')
datas += copy_metadata('safetensors')
datas += copy_metadata('scikit-image')
datas += copy_metadata('scipy')
datas += copy_metadata('setuptools')
datas += copy_metadata('shapely')
datas += copy_metadata('shellingham')
datas += copy_metadata('shiboken6')
datas += copy_metadata('six')
datas += copy_metadata('tifffile')
datas += copy_metadata('tqdm')
datas += copy_metadata('typer')
datas += copy_metadata('typing_extensions')
datas += copy_metadata('typing-inspection')
datas += copy_metadata('tzdata')
datas += copy_metadata('ujson')
datas += copy_metadata('urllib3')
datas += copy_metadata('wcwidth')
datas += copy_metadata('wheel')
datas += copy_metadata('wordsegment')
binaries += collect_dynamic_libs('paddle')
binaries += collect_dynamic_libs('paddlex')


a = Analysis(
    ['C:\\Users\\yao\\Projects\\video-subtitle-extractor\\gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=['paddle', 'paddleocr', 'paddlex', 'paddlex.utils', 'paddlex.utils.deps', 'paddlex.core', 'paddlex.core.predictor', 'paddle.dataset', 'paddle.dataset.*', 'paddle.fluid', 'paddle.fluid.core', 'paddle.framework', 'paddle.base', 'paddle.base.core', 'paddle.inference', 'paddle.utils', 'paddle.utils.try_import', 'cv2', 'PySide6', 'PySide6.QtWidgets', 'PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtNetwork', 'PySide6.QtOpenGL', 'PySide6.QtSvg', 'qfluentwidgets', 'qframelesswindow', 'qframelesswindow.utils', 'numpy', 'shapely', 'pyclipper', 'lmdb', 'skimage', 'PIL', 'tqdm', 'pysrt', 'Levenshtein', 'wordsegment', 'six', 'je_showinfilemanager', 'imageio_ffmpeg', 'backend', 'backend.main', 'backend.config', 'backend.bean', 'backend.bean.subtitle_area', 'backend.tools', 'backend.tools.constant', 'backend.tools.ocr', 'backend.tools.subtitle_ocr', 'backend.tools.subtitle_detect', 'backend.tools.hardware_accelerator', 'backend.tools.paddle_model_config', 'backend.tools.process_manager', 'backend.tools.reformat', 'backend.tools.theme_listener', 'backend.tools.version_service', 'backend.tools.python_runner', 'backend.tools.concurrent', 'backend.tools.concurrent.future', 'backend.tools.concurrent.task', 'backend.tools.concurrent.task_manager', 'ui', 'ui.home_interface', 'ui.advanced_setting_interface', 'ui.timeline_sync_interface', 'ui.component', 'ui.component.task_list_component', 'ui.component.video_display_component', 'ui.icon', 'ui.icon.my_fluent_icon'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'tkinter', 'IPython', 'jupyter', 'notebook', 'pytest', 'sphinx'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VideoSubtitleExtractor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['C:\\Users\\yao\\Projects\\video-subtitle-extractor\\design\\vse.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VideoSubtitleExtractor',
)
