[简体中文](README.md) | English

## Introduction

![License](https://img.shields.io/badge/License-Apache%202-red.svg)
![python version](https://img.shields.io/badge/Python-3.8+-blue.svg)
![support os](https://img.shields.io/badge/OS-Windows/macOS/Linux-green.svg)

**Video-subtitle-extractor** (vse) is a free, open-source tool which can help you rip the hard-coded subtitles from videos and automatically generate corresponding **srt** files for each video.  It includes the following implementations:

- Detect and extract subtitle frames (using traditional graphic method)
- Detect subtitle areas (i.e., coordinates) (as well as scene text if you want) (using deep learning algorithms)
- Converting graphic text into plain-text (using deep learning algorithms)
- Filter non-subtitle text (e.g., Logo and watermark etc.)
- Remove watermark, logo text and original video hard subtitles, see: [video-subtitle-remover (VSR)](https://github.com/YaoFANGUK/video-subtitle-remover/tree/main).
- Remove duplicated subtitle line and **generate srt file** (by calculating text similarity)
- Batch extraction. You can select multiple video files at one time and this tool can generate subtitles for each video.
- Multiple language support. You can extract subtitles in 87 languages such as: **Simplified Chinese**, **English**, 
  **Japanese**, **Korean**, **Arabic**, **Traditional Chinese**, **French**, **German**, **Russian**, **Spanish**, 
  **Portuguese**, **Italian**
- Multiple mode:
  - **fast**: use a lightweight model to extract subtitles quickly, but you may lose subtitles (**Recommended**).
  - **auto**: automatically judge the model, and use the lightweight model under CPU; Using accurate model under GPU, the speed of subtitle extraction is slow and almost no subtitles are lost.
  - **accurate**: use accurate model, detect subtitles frame by frame under GPU, and do not lose subtitles, but the speed is very slow (not recommended).

**Download**：

- Windows executable (might be slow when initial start): <a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/2.0.0/vse.exe">vse.exe</a> 

- Windows GPU version：<a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/2.0.0/vse_windows_gpu_v2.0.0.7z">vse_windows_gpu_v2.0.0.7z</a>

- Windows CPU version：<a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/2.0.0/vse_windows_cpu_v2.0.0.zip">vse_windows_cpu_v2.0.0.zip</a>

- MacOS：<a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/0.1.0/vse_macOS_CPU.dmg">vse_macOS_CPU.dmg</a>

## Features

- You don't need to do any preprocessing (e.g., binarization) and don't need to consider all aspects like subtitle fonts and size etc..
- This is an offline project. There is no online API call and you dont need to connect to the Internet service provider in order to get results. 
- For Command Line Interface(CLI) version, you can escape the subtitle area setting. This program will automatically detect the subtitle area for you.
- GPU support is available. You can install CUDA and cuDNN to speed up the detection and recognition process and even get more accurate results.

<p style="text-align:center;"><img src="https://github.com/YaoFANGUK/video-subtitle-extractor/raw/main/design/demo.png" alt="demo.png"/></p>

> **Provide your suggestions to improve this project in ISSUES**


## Demo

- Graphic User Interface (GUI):

<p style="text-align:center;"><img src="https://github.com/YaoFANGUK/video-subtitle-extractor/raw/main/design/demo.gif" alt="demo.gif"/></p>


- Command Line Interface (CLI): 

[![Demo Video](https://s1.ax1x.com/2020/10/05/0JWVeJ.png)](https://www.bilibili.com/video/BV1t5411h78J "Demo Video")


## Running Online

- **Google Colab Notebook with free GPU**: <a href="https://colab.research.google.com/github/YaoFANGUK/video-subtitle-extractor/blob/main/google_colab_en.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

> PS: can only run CLI version on Google Colab


## Getting Started with Source Code 

#### 1. Download and Install Miniconda 

- Windows: <a href="https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Windows-x86_64.exe">Miniconda3-py38_4.11.0-Windows-x86_64.exe</a>

- MacOS：<a href="https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-MacOSX-x86_64.pkg">Miniconda3-py38_4.11.0-MacOSX-x86_64.pkg</a>

- Linux: <a href="https://repo.anaconda.com/miniconda/Miniconda3-py38_4.11.0-Linux-x86_64.sh">Miniconda3-py38_4.11.0-Linux-x86_64.sh</a>


#### 2. Activate Vitrual Environment

(1) Switch to working directory
```shell
cd <your source code path>
```

(2) create and activate conda environment
```shell
conda create -n videoEnv python=3.8 pip
```

```shell
conda activate videoEnv  
```


#### 3. Install Dependencies

Before you install dependencies, make sure your python 3.8+ has installed as well as conda virtual environment has created and activated.

- For users who have CPU only (including Mac users): 

  - Install dependencies：

    ```shell
    pip install -r requirements.txt
    ```


- For users who have **NVIDIA** graphic card： **GPU version can achieve better accuracy**

  - Install **CUDA** and **cuDNN**

      <details>
          <summary>Linux</summary>
          <h5>(1) Download CUDA 11.7</h5>
          <pre><code>wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run</code></pre>
          <h5>(2) Install CUDA 11.7</h5>
          <pre><code>sudo sh cuda_11.7.0_515.43.04_linux.run</code></pre>
          <p>1. Input accept</p>
          <img src="https://i.328888.xyz/2023/03/31/iwVoeH.png" width="500" alt="">
          <p>2. make sure CUDA Toolkit 11.7 is chosen (If you have already installed driver, do not select Driver)</p>
          <img src="https://i.328888.xyz/2023/03/31/iwVThJ.png" width="500" alt="">
          <p>3. Add environment variables</p>
          <p>add the following content in  <strong>~/.bashrc</strong></p>
          <pre><code># CUDA
      export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
      export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}</code></pre>
          <p>Make sure it works</p>
          <pre><code>source ~/.bashrc</code></pre>
          <h5>(3) Download cuDNN 8.4.1</h5>
          <p><a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/1.0.0/cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz">cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz</a></p>
          <h5>(4) Install cuDNN 8.4.1</h5>
          <pre><code> tar -xf cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive.tar.xz
     mv cudnn-linux-x86_64-8.4.1.50_cuda11.6-archive cuda
     sudo cp ./cuda/include/* /usr/local/cuda-11.7/include/
     sudo cp ./cuda/lib/* /usr/local/cuda-11.7/lib64/
     sudo chmod a+r /usr/local/cuda-11.7/lib64/*
     sudo chmod a+r /usr/local/cuda-11.7/include/*</code></pre>
      </details>

      <details>
          <summary>Windows</summary>
          <h5>(1) Download CUDA 11.7</h5>
          <a href="https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_516.01_windows.exe">cuda_11.7.0_516.01_windows.exe</a>
          <h5>(2) Install CUDA 11.7</h5>
          <h5>(3) Download cuDNN 8.2.4</h5>
          <p><a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/1.0.0/cudnn-windows-x64-v8.2.4.15.zip">cudnn-windows-x64-v8.2.4.15.zip</a></p>
          <h5>(4) Install cuDNN 8.2.4</h5>
          <p>
             unzip "cudnn-windows-x64-v8.2.4.15.zip", then move all files in "bin, include, lib" in cuda 
      directory to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\
          </p>
      </details>


  - Install paddlepaddle:
    - windows:

      ```shell
        python -m pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
      ```

    - Linux:

      ```shell
        python -m pip install paddlepaddle-gpu==2.4.2.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
      ```

    > If you installed cuda 10.2，please install cuDNN 7.6.5 instead of cuDNN v8.x

    > If you installed cuda 11.2, please install cuDNN 8.1.1. However, RTX 30xx might be incompatible with cuda 11.2

  - Install other dependencies:

    ```shell
    pip install -r requirements_gpu.txt
    ```



#### 3. Running the program

- Run GUI version

```shell
python gui.py
```

- Run CLI version

```shell    
python ./backend/main.py
```

## Q & A

#### 1. Running Failure or Environment Problem 

Solution: If you are using a nvidia ampere architecture graphic card such as RTX 3050/3060/3070/3080, please use the latest PaddlePaddle version and CUDA 11.6 with cuDNN 8.2.1. Otherwise, check your which cuda and cudnn works with your GPU and then install them.

  
#### 2. For Windows users, if you encounter errors related to "geos_c.dll"

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


#### 3. How to generate executables
Using Nuitka version 0.6.19, copy all the files of ```site-packages``` under the Lib folder of the conda virtual environment to the ```dependencies``` folder, and comment all codes relevant to ```subprocess``` of ```image.py``` under the ```paddle``` library dataset, and use the following packaging command:

```shell
 python -m nuitka --standalone --mingw64 --include-data-dir=D:\vse\backend=backend --include-data-dir=D:\vse\dependencies=dependencies  --nofollow-imports --windows-icon-from-ico=D:\vse\design\vse.ico --plugin-enable=tk-inter,multiprocessing --output-dir=out .\gui.py
```

Make a single ```.exe``` file, (pip install zstandard can compress the file):

```shell
 python -m nuitka --standalone --windows-disable-console --mingw64 --lto no --include-data-dir=D:\vse\backend=backend --include-data-dir=D:\vse\dependencies=dependencies  --nofollow-imports --windows-icon-from-ico=D:\vse\design\vse.ico --plugin-enable=tk-inter,multiprocessing --output-dir=out --onefile .\gui.py
```


## Community Support

#### Jetbrains All Products Pack
The IDE this project used is supported by Jetbrains
<div align=center>
  <a href="https://jb.gg/OpenSourceSupport"><img src="https://resources.jetbrains.com/storage/products/company/brand/logos/jb_beam.png" alt="JetBrains Logo (Main) logo." width="80"></a>
</div>

