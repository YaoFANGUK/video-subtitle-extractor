## 项目特色

- 提取视频中的字幕，生成字幕文件，将水印(台标)等文本信息去除
- 采用PaddleOCR，无需设置调用任何API，不需要接入百度、阿里等OCR服务即可本地完成文本识别
- 无需用户手动设置字幕区域，该项目引入Paddle文本检测模型自动检测字幕区域

## 使用说明

- 下载项目文件，将工作目录切换到项目文件所在目录

```shell
cd video-subtitle-extractor
```


#### 1. 下载安装Anaconda 

<a href="https://www.anaconda.com/products/individual">https://www.anaconda.com/products/individual#Downloads</a>

#### 2. 使用conda创建项目虚拟环境并激活环境 (建议创建虚拟环境运行，也可以不用conda)

```shell
conda create --name videoEnv python=3.7
conda activate videoEnv  
```

#### 3. 使用pip安装依赖文件

- mac用户, cpu用户: 

```shell
pip install -r requirements.txt
```

- gpu用户：(使用cuda版本10.2)

```shell
pip install -r requirements(gpu).txt
```

#### 4. 运行程序

> 对于gpu用户：
> 请将config.py中第48行 USE_GPU = False 改为：


```python
USE_GPU = True
```

> 运行

```shell
python main.py
```

## 演示视频

[![Demo Video](https://s1.ax1x.com/2020/10/05/0JWVeJ.png)](https://www.bilibili.com/video/BV1t5411h78J "Demo Video")

