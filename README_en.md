[简体中文](README.md) | English

## Introduction

video-subtitle-extractor is used to extract hard-coded subtitles and generate **srt** file.  It includes the following implementations:

- detect and extract subtitle frames (and keyframes using traditional graphic method)
- detect the position (i.e. coordinates) of subtitles (as well as scene text if you want) (using deep learning method)
- recognise the content of subtitles (i.e. converting graphic text into plain-text) (using deep learning method)
- filter non-subtitle text
- remove duplicated subtitle line
- generate srt file



## Features

- You don't need to do any preprocessing to get an ideal result.
- This is an offline project. You don't need to make any API call from Internet service provider in order to get results. 
- For Command Line Interface(CLI) version, you don't need to manually set the location of subtitle. This program will automatically detect the subtitle area for you.
- GPU support is available. You can install CUDA and cuDNN to speed up the detection and recognition process and even get more accurate results.



## Demo

- Graphic User Interface:

<img src="https://z3.ax1x.com/2021/04/09/cNrA1A.png">

<div align="center">
  <img src="demo.gif"/>
</div>
- Command Line Interface: 

[![Demo Video](https://s1.ax1x.com/2020/10/05/0JWVeJ.png)](https://www.bilibili.com/video/BV1t5411h78J "Demo Video")



## Getting Started 

#### 1. (Optional) Download and Install Anaconda 

<a href="https://www.anaconda.com/products/individual">https://www.anaconda.com/products/individual#Downloads</a>

#### 2. (Optional) create and activate a conda virtual environment

```shell
conda create --name videoEnv python=3.7
conda activate videoEnv  
```

#### 3. Install dependencies via pip

make sure you have python 3.7 installed

- for macOS users and users who have CPU only: 

```shell
pip install -r requirements.txt
```

- for users who have Nvidia graphic card:

```shell
pip install -r requirements_gpu.txt
```

#### 4. Running the program

- Run GUI version

```shell
python gui.py
```

- Run CLI version

```shell    
python main.py
```



## Run on Google Colab

1. <a href="https://colab.research.google.com/" target="_blank">Login Colab</a>
2. Click "File", "Open Notebook", "GitHub" and then copy the following link:

https://github.com/YaoFANGUK/video-subtitle-extractor/blob/main/google_colab.ipynb

<img src="https://z3.ax1x.com/2021/03/30/ciG7Ps.png">   



## Debug

- For Windows users, if you encounter errors related to "geos_c.dll"

```text
    _lgeos = CDLL(os.path.join(sys.prefix, 'Library', 'bin', 'geos_c.dll'))
  File "C:\Users\Flavi\anaconda3\envs\subEnv\lib\ctypes\__init__.py", line 364, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: [WinError 126] The specified module could not be found。
```

Solution：

1) Uninstall Shapely

```shell
pip uninstall Shapely -y
```

2) Reinstall Shapely via conda (make sure you have anaconda or miniconda installed)

```shell
conda install Shapely             
```

