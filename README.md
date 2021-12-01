简体中文 | [English](README_en.md)

## 项目简介

![License](https://img.shields.io/badge/License-Apache%202-red.svg)
![python version](https://img.shields.io/badge/Python-3.8+-blue.svg)
![support os](https://img.shields.io/badge/OS-Windows/macOS/Linux-green.svg)

video-subtitle-extractor是一款将视频中的硬字幕提取为外挂字幕文件(srt格式)的软件。
主要实现了以下功能：

- 提取视频中的关键帧
- 检测视频帧中文本的所在位置
- 识别视频帧中文本的内容
- 过滤非字幕区域的文本
- 去除重复字幕行，去除水印(台标)文本
- 生成srt字幕文件
- 多语言：支持**中文/英文**、**繁体中文**、**日语**、**韩语**、**法语**、**德语**字幕的提取
- 多模式：
  - **快速** - 快速提取字幕但可能丢字幕
  - **精准** - 不丢字幕但速度较慢

QQ交流群：210150985

**下载地址**：

- Windows GPU版本：
  - 国内：<a href="https://github.91chi.fun/https://github.com//YaoFANGUK/video-subtitle-extractor/releases/download/0.2.0/vse_windows_GPU.7z">vse_windows_GPU.7z</a>
  - 国外：<a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/0.2.0/vse_windows_GPU.7z">vse_windows_GPU.7z</a>

- Windows CPU版本：
  - 国内：<a href=https://github.91chi.fun//https://github.com//YaoFANGUK/video-subtitle-extractor/releases/download/0.2.0/vse_windows_CPU.zip>vse_windows_CPU.zip</a>
  - 国外：<a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/0.2.0/vse_windows_CPU.zip">vse_windows_CPU.zip</a> 

- MacOS CPU版本：
  - 国内：<a href="https://github.91chi.fun/https://github.com//YaoFANGUK/video-subtitle-extractor/releases/download/0.1.0/vse_macOS_CPU.dmg">vse_macOS_CPU.dmg</a>
  - 国外：<a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/0.1.0/vse_macOS_CPU.dmg">vse_macOS_CPU.dmg</a> 



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

<p style="text-align:center;"><img src="design/demo.gif" alt="demo.gif"/></p>

- 点击查看视频教程 👇

[![GPU版本安装教程](https://z3.ax1x.com/2021/09/08/h7hhNV.png)](https://www.bilibili.com/video/bv11L4y1Y7Tj "GUP版本安装教程")



## 在线运行

- 使用**Google Colab Notebook**(免费GPU): <a href="https://colab.research.google.com/github/YaoFANGUK/video-subtitle-extractor/blob/main/google_colab.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

> PS: Google Colab只能运行CLI版本



## 使用说明

#### 1. 下载安装Anaconda 

<a href="https://www.anaconda.com/products/individual">https://www.anaconda.com/products/individual#Downloads</a>

#### 2. 使用conda创建项目虚拟环境并激活环境 (建议创建虚拟环境运行，以免后续出现问题)

```shell
conda create --name videoEnv python=3.8
```

```shell
conda activate videoEnv  
```

#### 3. 安装依赖文件

请确保你已经安装 python 3.8+

- CPU用户 (mac用户) : 

  - 安装依赖：

    ```shell
    pip install -r requirements.txt
    ```

- GPU用户(有N卡)： **要达到高精度的识别率请使用GPU版**

  - 安装CUDA 11.2和cuDNN 8.1.1

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
        <p>国内：<a href="https://github.91chifun.workers.dev/https://github.com//YaoFANGUK/video-subtitle-extractor/releases/download/0.2.0/cudnn-11.2-linux-x64-v8.1.1.33.tgz">cudnn-11.2-linux-x64-v8.1.1.33.tgz</a></p>
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
          <h5>(1) 下载CUDA 11.2</h5>
          <a href="https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.89_win10.exe">cuda_11.2.0_460.89_win10.exe</a>
          <h5>(2) 安装CUDA 11.2</h5>
          <h5>(3) 下载cuDNN 8.1.1</h5>
          <p>国内：<a href="https://github.91chifun.workers.dev/https://github.com//YaoFANGUK/video-subtitle-extractor/releases/download/0.2.0/cudnn-11.2-windows-x64-v8.1.1.33.zip">cudnn-11.2-windows-x64-v8.1.1.33.zip</a></p>
          <p>国外：<a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/0.2.0/cudnn-11.2-windows-x64-v8.1.1.33.zip">cudnn-11.2-windows-x64-v8.1.1.33.zip</a></p>
          <h5>(4) 安装cuDNN 8.1.1</h5>
          <p>
             将cuDNN解压后的cuda文件夹中的bin, include, lib目录下的文件复制到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\对应目录下
          </p>
      </details>

  - 安装paddlepaddle:

    ```shell
    conda install paddlepaddle-gpu==2.1.3 cudatoolkit=11.2 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/Paddle/ -c conda-forge 
    ```

    > 如果安装cuda 10.2，请对应安装7.6.5的cuDNN, **请不要使用cuDNN v8.x 和 cuda 10.2的组合**

  - 安装其他依赖:

    ```shell
    pip install -r requirements_gpu.txt
    ```

#### 4. 运行程序

- 运行图形化界面版本(GUI)

```shell
python gui.py
```

- 运行命令行版本(CLI)

```shell    
python main.py
```



## 常见问题与解决方案

#### 1. 运行不正常/没有结果

解决方案：如果你使用的是3060/3070/3080等安培架构的显卡，请使用最新的paddlepaddle版本以及最新的cuda 11+


#### 2. CondaHTTPError

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
