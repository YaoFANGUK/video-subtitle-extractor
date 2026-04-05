"""
PyInstaller 打包脚本
用法：
    python build.py              # 打包 CPU 版本
    python build.py --cuda       # 打包 CUDA 版本（收集 NVIDIA 相关二进制）
"""
import os
import sys
import subprocess
import importlib.metadata

def get_package_data():
    """获取需要 --collect-data 的包列表"""
    data_packages = [
        "paddleocr",
        "paddlex",
        "paddle",
        "qfluentwidgets",
        "PySide6",
        "wordsegment",
    ]
    return data_packages


def get_binary_packages():
    """获取需要 --collect-binaries 的包列表"""
    return ["paddle", "paddlex"]


def get_metadata_packages():
    """获取需要 --copy-metadata 的包列表（运行时通过 importlib.metadata 查找的依赖）"""
    # paddlex 在运行时通过 importlib.metadata.version() 检查几乎所有依赖
    # 所以最安全的做法是收集所有已安装包的 metadata
    all_installed = []
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            all_installed.append(name)
    return all_installed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="VSE PyInstaller 打包脚本")
    parser.add_argument("--cuda", action="store_true", help="包含 NVIDIA CUDA 和 cuDNN 依赖")
    parser.add_argument("--onedir", action="store_true", help="打包为目录模式（默认单文件）")
    parser.add_argument("--noconfirm", action="store_true", help="覆盖已有输出目录")
    args = parser.parse_args()

    work_dir = os.path.dirname(os.path.abspath(__file__))
    main_file = os.path.join(work_dir, "gui.py")
    icon_path = os.path.join(work_dir, "design", "vse.ico")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        main_file,
        "--name", "VideoSubtitleExtractor",
    ]

    # 单文件 or 目录模式
    if not args.onedir:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    # 窗口模式（无控制台）
    cmd.append("--windowed")

    # 图标
    if os.path.exists(icon_path):
        cmd += ["--icon", icon_path]

    # 收集数据文件
    for pkg in get_package_data():
        cmd += ["--collect-data", pkg]

    # 收集二进制
    for pkg in get_binary_packages():
        cmd += ["--collect-binaries", pkg]

    if args.cuda:
        cmd += ["--collect-binaries", "nvidia"]
        cmd += ["--collect-binaries", "nvidia.cublas"]
        cmd += ["--collect-binaries", "nvidia.cudnn"]
        cmd += ["--collect-binaries", "nvidia.cuda_runtime"]

    # 复制元数据
    for pkg in get_metadata_packages():
        cmd += ["--copy-metadata", pkg]

    # 项目自带的非 Python 数据文件
    # backend/interface/*.ini (翻译文件)
    cmd += ["--add-data", os.path.join(work_dir, "backend", "interface") + os.pathsep + "backend/interface"]
    # backend/configs
    cmd += ["--add-data", os.path.join(work_dir, "backend", "configs") + os.pathsep + "backend/configs"]
    # backend/models (OCR 模型)
    cmd += ["--add-data", os.path.join(work_dir, "backend", "models") + os.pathsep + "backend/models"]
    # backend/subfinder (VideoSubFinder 二进制)
    cmd += ["--add-data", os.path.join(work_dir, "backend", "subfinder") + os.pathsep + "backend/subfinder"]
    # design 目录（图标等）
    cmd += ["--add-data", os.path.join(work_dir, "design") + os.pathsep + "design"]
    # ui/icon SVG 图标
    cmd += ["--add-data", os.path.join(work_dir, "ui", "icon") + os.pathsep + "ui/icon"]
    # 字体文件
    font_file = os.path.join(work_dir, "backend", "tools", "NotoSansCJK-Bold.otf")
    if os.path.exists(font_file):
        cmd += ["--add-data", font_file + os.pathsep + "backend/tools"]

    # 运行时路径修复：确保 _MEIPASS 下能找到项目模块
    cmd += ["--runtime-tmpdir", "."]

    # 隐式导入（PyInstaller 可能检测不到的）
    hidden_imports = [
        "paddle",
        "paddleocr",
        "paddlex",
        "paddlex.utils",
        "paddlex.utils.deps",
        "paddlex.core",
        "paddlex.core.predictor",
        "paddle.dataset",
        "paddle.dataset.*",
        "paddle.fluid",
        "paddle.fluid.core",
        "paddle.framework",
        "paddle.base",
        "paddle.base.core",
        "paddle.inference",
        "paddle.utils",
        "paddle.utils.try_import",
        "cv2",
        "PySide6",
        "PySide6.QtWidgets",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtNetwork",
        "PySide6.QtOpenGL",
        "PySide6.QtSvg",
        "qfluentwidgets",
        "qframelesswindow",
        "qframelesswindow.utils",
        "numpy",
        "shapely",
        "pyclipper",
        "lmdb",
        "skimage",
        "PIL",
        "tqdm",
        "pysrt",
        "Levenshtein",
        "wordsegment",
        "six",
        "je_showinfilemanager",
        "imageio_ffmpeg",
        "backend",
        "backend.main",
        "backend.config",
        "backend.bean",
        "backend.bean.subtitle_area",
        "backend.tools",
        "backend.tools.constant",
        "backend.tools.ocr",
        "backend.tools.subtitle_ocr",
        "backend.tools.subtitle_detect",
        "backend.tools.hardware_accelerator",
        "backend.tools.paddle_model_config",
        "backend.tools.process_manager",
        "backend.tools.reformat",
        "backend.tools.theme_listener",
        "backend.tools.version_service",
        "backend.tools.python_runner",
        "backend.tools.concurrent",
        "backend.tools.concurrent.future",
        "backend.tools.concurrent.task",
        "backend.tools.concurrent.task_manager",
        "ui",
        "ui.home_interface",
        "ui.advanced_setting_interface",
        "ui.timeline_sync_interface",
        "ui.component",
        "ui.component.task_list_component",
        "ui.component.video_display_component",
        "ui.icon",
        "ui.icon.my_fluent_icon",
    ]
    for imp in hidden_imports:
        cmd += ["--hidden-import", imp]

    # 排除不需要的大包以减小体积
    excludes = [
        "matplotlib",
        "tkinter",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "sphinx",
    ]
    for exc in excludes:
        cmd += ["--exclude-module", exc]

    if args.noconfirm:
        cmd.append("--noconfirm")

    print("=" * 60)
    print("PyInstaller 打包命令：")
    print(" ".join(cmd))
    print("=" * 60)

    try:
        subprocess.run(cmd, check=True)
        print("\n打包完成！输出目录：", os.path.join(work_dir, "dist"))
    except subprocess.CalledProcessError as e:
        print("\n打包失败：", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
