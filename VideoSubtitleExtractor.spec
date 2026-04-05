# -*- mode: python ; coding: utf-8 -*-
import os
import sys
import importlib

block_cipher = None

# Project root
ROOT = os.path.abspath(SPECPATH)

# --- Data files to bundle ---
datas = []

# 1. UI translation files
datas.append((os.path.join(ROOT, 'backend', 'interface'), 'backend/interface'))

# 2. OCR model files
datas.append((os.path.join(ROOT, 'backend', 'models'), 'backend/models'))

# 3. Design assets (icons, etc.)
datas.append((os.path.join(ROOT, 'design'), 'design'))

# 4. Config directory
datas.append((os.path.join(ROOT, 'config'), 'config'))

# 5. VideoSubFinder macOS binaries
datas.append((os.path.join(ROOT, 'backend', 'subfinder', 'macos'), 'backend/subfinder/macos'))

# 6. UI files
datas.append((os.path.join(ROOT, 'ui'), 'ui'))

# 6.5. Backend configs (typoMap.json etc.)
backend_configs = os.path.join(ROOT, 'backend', 'configs')
if os.path.exists(backend_configs):
    datas.append((backend_configs, 'backend/configs'))

# 6.6. Backend tools data files (fonts etc.)
backend_tools_dir = os.path.join(ROOT, 'backend', 'tools')
for f in os.listdir(backend_tools_dir):
    if f.endswith(('.otf', '.ttf', '.ttc', '.json', '.yml', '.yaml', '.cfg', '.txt')):
        datas.append((os.path.join(backend_tools_dir, f), os.path.join('backend', 'tools')))

# 6.7. Package metadata for runtime dependency checks (paddlex uses importlib.metadata)
site_packages = '/opt/miniconda3/envs/vse/lib/python3.13/site-packages'
_paddlex_deps = [
    'pyclipper-1.4.0.dist-info',
    'shapely-2.1.2.dist-info',
    'opencv_contrib_python-4.10.0.84.dist-info',
    'pypdfium2-5.6.0.dist-info',
    'scikit_image-0.26.0.dist-info',
    'scipy-1.17.1.dist-info',
    'lmdb-1.8.1.dist-info',
    'pillow-10.4.0.dist-info',
    'numpy-2.4.4.dist-info',
    'levenshtein-0.26.0.dist-info',
    'wordsegment-1.3.1.dist-info',
    'pysrt-1.1.2.dist-info',
    'cython-3.2.4.dist-info',
    'packaging-26.0.dist-info',
    'paddleocr-3.4.0.dist-info',
    'paddlepaddle-3.3.1.dist-info',
    'paddlex-3.4.3.dist-info',
    'opencv_python-4.13.0.92.dist-info',
    'opencv_python_headless-4.13.0.92.dist-info',
]
for meta in _paddlex_deps:
    meta_path = os.path.join(site_packages, meta)
    if os.path.exists(meta_path):
        datas.append((meta_path, meta))

# 7. wordsegment data files (unigrams.txt etc.)
wordsegment_dir = os.path.join(site_packages, 'wordsegment')
if os.path.exists(wordsegment_dir):
    for f in os.listdir(wordsegment_dir):
        if f.endswith(('.txt', '.py', '.json')):
            datas.append((os.path.join(wordsegment_dir, f), 'wordsegment'))

# 8. PaddleOCR + paddlex data files (official recommendation: --collect-data paddlex)
paddlex_dir = os.path.join(site_packages, 'paddlex')
if os.path.exists(paddlex_dir):
    datas.append((paddlex_dir, 'paddlex'))
paddleocr_dir = os.path.join(site_packages, 'paddleocr')
if os.path.exists(paddleocr_dir):
    datas.append((paddleocr_dir, 'paddleocr'))

# 9. PySide6 Qt plugins (platforms, imageformats, styles, iconengines)
pyside6_dir = os.path.dirname(importlib.import_module('PySide6').__file__)
qt_plugins = os.path.join(pyside6_dir, 'Qt', 'plugins')
for plugin_dir in ['platforms', 'styles', 'imageformats', 'iconengines']:
    src = os.path.join(qt_plugins, plugin_dir)
    if os.path.exists(src):
        datas.append((src, os.path.join('PySide6', 'Qt', 'plugins', plugin_dir)))

# PySide6 Qt lib dylibs
qt_lib = os.path.join(pyside6_dir, 'Qt', 'lib')
if os.path.exists(qt_lib):
    datas.append((qt_lib, os.path.join('PySide6', 'Qt', 'lib')))

# PySide6 resources
qt_resources = os.path.join(pyside6_dir, 'Qt', 'resources')
if os.path.exists(qt_resources):
    datas.append((qt_resources, os.path.join('PySide6', 'Qt', 'resources')))

# 8. Paddle native libraries
paddle_dir = os.path.dirname(importlib.import_module('paddle').__file__)
paddle_libs = os.path.join(paddle_dir, 'libs')
if os.path.exists(paddle_libs):
    datas.append((paddle_libs, os.path.join('paddle', 'libs')))
# Paddle core C++ engine (228MB, lives outside paddle/libs/)
paddle_base = os.path.join(paddle_dir, 'base')
if os.path.exists(paddle_base):
    datas.append((paddle_base, os.path.join('paddle', 'base')))

# 9. PaddleOCR package resources
paddleocr_dir = os.path.dirname(importlib.import_module('paddleocr').__file__)
for subdir in os.listdir(paddleocr_dir):
    full = os.path.join(paddleocr_dir, subdir)
    if os.path.isdir(full) and not subdir.startswith('__'):
        # Include ppocr, ppstructure etc. data dirs
        for sub_root, sub_dirs, sub_files in os.walk(full):
            for f in sub_files:
                if f.endswith(('.txt', '.json', '.yml', '.yaml', '.cfg')):
                    rel = os.path.relpath(sub_root, paddleocr_dir)
                    datas.append((sub_root, os.path.join('paddleocr', rel)))

# 10. Cython Utility files (needed by paddle.utils.cpp_extension)
cython_dir = os.path.dirname(importlib.import_module('Cython').__file__)
cython_util = os.path.join(cython_dir, 'Utility')
if os.path.exists(cython_util):
    datas.append((cython_util, os.path.join('Cython', 'Utility')))

# --- Hidden imports ---
hiddenimports = [
    'paddle',
    'paddle.fluid',
    'paddle.dataset',
    'paddleocr',
    'paddleocr.ppocr',
    'paddleocr.ppocr.db',
    'paddleocr.ppocr.crnn',
    'paddleocr.ppocr.cls',
    'paddleocr.ppocr.utils',
    'paddleocr.paddleocr',
    'cv2',
    'PySide6',
    'PySide6.QtCore',
    'PySide6.QtWidgets',
    'PySide6.QtGui',
    'qfluentwidgets',
    'qframelesswindow',
    'qframelesswindow.utils',
    'shapely',
    'shapely.geometry',
    'pyclipper',
    'lmdb',
    'Levenshtein',
    'wordsegment',
    'pysrt',
    'skimage',
    'skimage.filters',
    'skimage.measure',
    'showinfm',
    'imageio_ffmpeg',
    'numpy',
    'PIL',
    'tqdm',
    'six',
    'configparser',
    'packaging',
    'multiprocessing',
    'queue',
    'threading',
    'json',
    're',
    'collections',
    'types',
    'Cython',
    'Cython.Utility',
]

# --- Binary / strip excludes ---
excludes = [
    'tkinter', 'matplotlib',
    'IPython', 'notebook', 'jupyter',
]

a = Analysis(
    ['gui.py'],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VideoSubtitleExtractor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    icon=os.path.join(ROOT, 'build', 'icon.icns'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='VideoSubtitleExtractor',
)

app = BUNDLE(
    coll,
    name='VideoSubtitleExtractor.app',
    icon=os.path.join(ROOT, 'build', 'icon.icns'),
    bundle_identifier='com.vse.subtitle-extractor',
    info_plist={
        'CFBundleName': 'VideoSubtitleExtractor',
        'CFBundleDisplayName': 'Video Subtitle Extractor',
        'CFBundleVersion': '2.2.0',
        'CFBundleShortVersionString': '2.2.0',
        'NSHighResolutionCapable': True,
        'LSMinimumSystemVersion': '10.15',
        'CFBundleDocumentTypes': [],
        'NSRequiresAquaSystemAppearance': False,
    },
)

# --- Post-build: re-sign app ---
import subprocess as sp
sp.run(['codesign', '--force', '--sign', '-', os.path.join(ROOT, 'dist', 'VideoSubtitleExtractor.app')], capture_output=True)
print('Post-build: re-signed app')