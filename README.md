简体中文 | [English](README_en.md)

## 项目简介

![License](https://img.shields.io/badge/License-Apache%202-red.svg)
![python version](https://img.shields.io/badge/Python-3.8+-blue.svg)
![support os](https://img.shields.io/badge/OS-Windows/macOS/Linux-green.svg)  

Video-subtitle-extractor (vse) 是一款将视频中的硬字幕提取为外挂字幕文件(srt格式)的软件。
主要实现了以下功能：

- 提取视频中的关键帧
- 检测视频帧中文本的所在位置
- 识别视频帧中文本的内容
- 过滤非字幕区域的文本，去除水印（台标）文本
- 去除重复字幕行，生成srt字幕文件
- 支持视频字幕**批量提取**
- 多语言：支持**中文/英文**、**日语**、**韩语**、**阿拉伯语**、**繁体中文**、**法语**、**德语**、**俄语**、**西班牙语**、**葡萄牙语**、**意大利语**字幕的提取
- 多模式：
  - **快速** - 快速提取字幕但可能丢字幕（推荐）
  - **精准** - 不丢字幕但速度较慢


**使用说明**：

- 有使用问题请加群讨论，QQ群：210150985

- 视频以及程序路径请**不要带中文和空格**，否则可能出现未知错误！！！

 > 如：以下存放视频和代码的路径都不行
 >
 > D:\下载\vse\运行程序.exe（路径含中文）
 >
 > E:\study\kaoyan\sanshang youya.mp4 （路径含空格） 

- 直接下载压缩包解压运行，如果不能运行再按照下面的教程，尝试源码安装conda环境运行

**下载地址**：

- Windows 单文件版本(双击直接运行，每次打开时会有一点慢，**推荐小白使用**)： <a href=https://github.91chi.fun/https://github.com//YaoFANGUK/video-subtitle-extractor/releases/download/1.0.0/vse.exe>vse.exe</a>

- Windows GPU版本： <a href="https://github.91chi.fun/https://github.com//YaoFANGUK/video-subtitle-extractor/releases/download/1.0.0/vse_windows_GPU.7z">vse_windows_GPU.7z</a>

- Windows CPU版本： <a href=https://github.91chi.fun/https://github.com//YaoFANGUK/video-subtitle-extractor/releases/download/1.0.0/vse_windows_CPU.zip>vse_windows_CPU.zip</a>

- MacOS CPU版本： <a href="https://github.91chi.fun/https://github.com//YaoFANGUK/video-subtitle-extractor/releases/download/0.1.0/vse_macOS_CPU.dmg">vse_macOS_CPU.dmg</a>

> PS: 若无法下载，请前往<a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases"> Release </a>下载


## 项目特色

- 采用本地进行OCR识别，无需设置调用任何API，不需要接入百度、阿里等在线OCR服务即可本地完成文本识别
- 支持GPU加速，GPU加速后可以获得更高的准确率与更快的提取速度
- (CLI版本) 无需用户手动设置字幕区域，项目通过文本检测模型自动检测字幕区域
- (GUI版本) 图形化界面

<img src="https://z3.ax1x.com/2021/04/09/cNrA1A.png" alt="demo">

点击【打开】后选择视频文件，调整字幕区域，点击【运行】

> **有任何改进意见请在ISSUES中提出**



## 演示

- GUI版：

<p style="text-align:center;"><img src="https://github.com/YaoFANGUK/video-subtitle-extractor/raw/main/design/demo.gif" alt="demo.gif"/></p>

- 点击查看视频教程 👇

[![GPU版本安装教程](https://s1.ax1x.com/2022/04/15/L3KzLR.png)](https://www.bilibili.com/video/bv11L4y1Y7Tj "GUP版本安装教程")



## 在线运行

- 使用**Google Colab Notebook**(免费GPU): <a href="https://colab.research.google.com/github/YaoFANGUK/video-subtitle-extractor/blob/main/google_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

> PS: Google Colab只能运行CLI版本



## 源码使用说明

#### 1. 下载安装Miniconda 

- Windows: <a href="https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Windows-x86_64.exe">Miniconda3-py38_4.11.0-Windows-x86_64.exe</a>


- MacOS：<a href="https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-MacOSX-x86_64.pkg">Miniconda3-py38_4.11.0-MacOSX-x86_64.pkg</a>


- Linux: <a href="https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh">Miniconda3-py38_4.11.0-Linux-x86_64.sh</a>

#### 2. 创建并激活虚机环境

```shell
conda create -n videoEnv python=3.8
```

```shell
conda activate videoEnv
```

#### 2. 安装依赖文件

请确保你已经安装 python 3.8+，使用conda创建项目虚拟环境并激活环境 (建议创建虚拟环境运行，以免后续出现问题)

- CPU用户 (Mac用户) : 

  - 安装依赖：
    ```shell
    pip install -r requirements.txt
    ```

- GPU用户(有N卡)： **要达到高精度的识别率请使用GPU版**
  
  - 安装CUDA和cuDNN

    <details>
        <summary>Linux用户</summary>
        <h5>(1) 下载CUDA 11.2</h5>
        <pre><code>wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run</code></pre>
        <h5>(2) 安装CUDA 11.2</h5>
        <pre><code>sudo sh cuda_11.2.0_460.27.04_linux.run --override</code></pre>
        <p>1. 输入accept</p>
        <img src="https://z3.ax1x.com/2021/05/24/gv0AVU.png" width="500" alt="">
        <p>2. 选中CUDA Toolkit 11.2（如果你没有安装nvidia驱动则选中Driver，如果你已经安装了nvidia驱动请不要选中driver），之后选中install，回车</p>
        <img src="https://z3.ax1x.com/2021/10/11/5VnwfH.png" width="500" alt="">
        <p>3. 添加环境变量</p>
        <p>在 ~/.bashrc 加入以下内容</p>
        <pre><code># CUDA
    export PATH=/usr/local/cuda-11.2/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}</code></pre>
        <p>使其生效</p>
        <pre><code>source ~/.bashrc</code></pre>
        <h5>(3) 下载cuDNN 8.1.1</h5>
        <p>国内：<a href="https://github.91chi.fun/https://github.com//YaoFANGUK/video-subtitle-extractor/releases/download/0.2.0/cudnn-11.2-linux-x64-v8.1.1.33.tgz">cudnn-11.2-linux-x64-v8.1.1.33.tgz</a></p>
        <p>国外：<a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/0.2.0/cudnn-11.2-linux-x64-v8.1.1.33.tgz">cudnn-11.2-linux-x64-v8.1.1.33.tgz</a></p>
        <h5>(4) 安装cuDNN 8.1.1</h5>
        <pre><code> tar -zxvf cudnn-11.2-linux-x64-v8.1.1.33.tgz
     sudo cp ./cuda/include/* /usr/local/cuda-11.2/include/
     sudo cp ./cuda/lib64/* /usr/local/cuda-11.2/lib64/
     sudo chmod a+r /usr/local/cuda-11.2/lib64/*
     sudo chmod a+r /usr/local/cuda-11.2/include/*</code></pre>
    </details>
  
    <details>
          <summary>Windows用户</summary>
          <h5>(1) 下载CUDA 11.6</h5>
          <a href="https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_511.65_windows.exe">cuda_11.6.2_511.65_windows.exe</a>
          <h5>(2) 安装CUDA 11.6</h5>
          <h5>(3) 下载cuDNN 8.2.1</h5>
          <p>国内：<a href="https://github.91chi.fun/https://github.com//YaoFANGUK/video-subtitle-extractor/releases/download/1.0.0/cudnn-11.3-windows-x64-v8.2.1.32.zip">cudnn-11.3-windows-x64-v8.2.1.32.zip</a></p>
          <p>国外：<a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/1.0.0/cudnn-11.3-windows-x64-v8.2.1.32.zip">cudnn-11.3-windows-x64-v8.2.1.32.zip</a></p>
          <h5>(4) 安装cuDNN 8.2.1</h5>
          <p>
             将cuDNN解压后的cuda文件夹中的bin, include, lib目录下的文件复制到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\对应目录下
          </p>
      </details>

  > 如果你使用的是3050/3060/3070/3080等安培架构的显卡，请使用最新的paddlepaddle版本以及最新的cuda 11.6

  - 安装paddlepaddle:

    - windows:

        ```shell
        python -m pip install paddlepaddle-gpu==2.2.2.post112 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html 
        ```

    - Linux:

        ```shell
        python -m pip install paddlepaddle-gpu==2.2.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html 
        ```

        > 如果安装cuda 10.2，请对应安装7.6.5的cuDNN, **请不要使用cuDNN v8.x 和 cuda 10.2的组合**
 
        > 如果安装cuda 11.2，请对应安装8.1.1的cuDNN, **30系列以上的显卡驱动可能不支持 cuda 11.2的安装**  


  - 安装其他依赖:

    ```shell
    pip install -r requirements_gpu.txt
    ```
  

#### 3. 运行程序

- 运行图形化界面版本(GUI)

```shell
python gui.py
```

- 运行命令行版本(CLI)

```shell
python ./backend/main.py
```



## 常见问题与解决方案

#### 1. 运行不正常/没有结果/cuda及cudnn问题

解决方案：根据自己的显卡型号、显卡驱动版本，安装对应的cuda与cudnn


#### 2. CondaHTTPError

将项目中的.condarc放在用户目录下(C:\Users\\<你的用户名>)，如果用户目录已经存在该文件则覆盖

解决方案：<a href="https://zhuanlan.zhihu.com/p/260034241">https://zhuanlan.zhihu.com/p/260034241</a>

#### 3. Windows下出现geos_c.dll错误

```text
    _lgeos = CDLL(os.path.join(sys.prefix, 'Library', 'bin', 'geos_c.dll'))
  File "C:\Users\Flavi\anaconda3\envs\subEnv\lib\ctypes\__init__.py", line 364, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: [WinError 126] 找不到指定的模块。
```

解决方案：

1) 卸载Shapely

```shell
pip uninstall Shapely -y
```

2) 使用conda重新安装Shapely

```shell
conda install Shapely             
```


#### 4. Nuitka打包代码闪退

使用Nuitka版本```0.6.19```，将conda虚拟环境Lib文件夹下site-packages的所有文件复制到dependencies文件夹中，把paddle库dataset下image.py的有关subprocess代码全部注释了，使用以下打包命令：

```shell
 python -m nuitka --standalone --mingw64 --include-data-dir=D:\vse\backend=backend --include-data-dir=D:\vse\dependencies=dependencies  --nofollow-imports --windows-icon-from-ico=D:\vse\design\vse.ico --plugin-enable=tk-inter,multiprocessing --output-dir=out .\gui.py
```

编译成单个文件（pip安装zstandard可以减小体积）
```shell
 python -m nuitka --standalone --windows-disable-console --mingw64 --lto no --include-data-dir=D:\vse\backend=backend --include-data-dir=D:\vse\dependencies=dependencies  --nofollow-imports --windows-icon-from-ico=D:\vse\design\vse.ico --plugin-enable=tk-inter,multiprocessing --output-dir=out --onefile .\gui.py
```



## 社区支持

#### Jetbrains 全家桶支持
本项目开发所使用的IDE由Jetbrains支持。
<div align=center>
  <a href="https://jb.gg/OpenSourceSupport"><img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jb_beam.png" alt="JetBrains Logo (Main) logo." width="80"></a>
</div>

