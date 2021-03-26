简体中文 | [English](README_en.md)

## 项目简介

video-subtitle-extractor是一款将视频中的硬字幕提取为外挂字幕文件(srt格式)的软件。
主要实现了以下功能：
- 提取视频中的关键帧
- 检测视频帧中文本的所在位置
- 识别视频视频帧中文本的内容
- 过滤非字幕区域的文本
- 去除重复字幕行，去除水印(台标)文本
- 生成srt字幕文件

## 项目特色

- 采用本地进行OCR识别，无需设置调用任何API，不需要接入百度、阿里等在线OCR服务即可本地完成文本识别
- 无需用户手动设置字幕区域，项目通过文本检测模型自动检测字幕区域

## 演示视频

[![Demo Video](https://s1.ax1x.com/2020/10/05/0JWVeJ.png)](https://www.bilibili.com/video/BV1t5411h78J "Demo Video")

## 使用说明

#### 1. (可选) 下载安装Anaconda 

<a href="https://www.anaconda.com/products/individual">https://www.anaconda.com/products/individual#Downloads</a>

#### 2. (可选) 使用conda创建项目虚拟环境并激活环境 (建议创建虚拟环境运行，也可以不用conda)

```shell
conda create --name videoEnv python=3.7
conda activate videoEnv  
```

#### 3. 使用pip安装依赖文件

如果没有安装Anaconda，请确保你已经安装 python 3.7

- mac用户, CPU用户: 

```shell
pip install -r requirements.txt
```

- 有N卡的GPU用户：(使用cuda版本10.2)

```shell
pip install -r requirements(gpu).txt
```

#### 4. 运行程序

对于**GPU用户**： 请将config.py中第48行 USE_GPU = False 改为：

```python
USE_GPU = True
```

运行

```shell
python main.py
```

