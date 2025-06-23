[简体中文](README.md) | English

## Introduction

![License](https://img.shields.io/badge/License-Apache%202-red.svg)
![python version](https://img.shields.io/badge/Python-3.12+-blue.svg)
![support os](https://img.shields.io/badge/OS-Windows/macOS/Linux-green.svg)

**Video-subtitle-extractor** (VSE) is a free, open-source tool which can help you rip the hard-coded subtitles from videos and automatically generate corresponding **srt** files for each video.  It includes the following implementations:

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
- Multi-mode:
  - **fast**: (Recommended) Uses a lightweight model for quick subtitle extraction, though it might miss a small amount of subtitles and contains a few typos.
  - **auto**: (Recommended) Automatically selects the model. It uses the lightweight model under the CPU, and the precise model under the GPU. While subtitle extraction speed is slower and might miss a minor amount of subtitles, there are almost no typos.
  - **accurate**: (Not Recommended) Uses the precise model with frame-by-frame detection under the GPU, ensuring no missed subtitles and almost non-existent typos, but the speed is **very slow**.

<p style="text-align:center;"><img src="https://github.com/YaoFANGUK/video-subtitle-extractor/raw/main/design/demo.png" alt="demo.png"/></p>

**Features**：

- You don't need to do any preprocessing (e.g., binarization) and don't need to consider all aspects like subtitle fonts and size etc..
- This is an offline project. There is no online API call and you dont need to connect to the Internet service provider in order to get results.

**Usage**：

- After clicking "Open", select video file(s), adjust the subtitle area, and then click "Run".
  - Single file extraction: When opening a file, choose a single video.
  - Batch extraction: When opening files, choose multiple videos, ensure that every video's resolution and subtitle area remain consistent.

- Remove watermark text/replace specific text:
> If specific text needs to be deleted from generated .srt file, or specific text needs to be replaced, you can edit the ``backend/configs/typoMap.json`` file and add the content you want to replace or remove.

```json
{
    "l'm": "I'm",
    "l just": "I just",
    "Let'sqo": "Let's go",
    "Iife": "life",
    "威筋": "threat",
    "性感荷官在线发牌": ""
}
```

> In this way, you can replace all occurrences of "威筋" in the text with "threat" and delete all instances of the text "性感荷官在线发牌".


- Directly download the compressed package, unzip it and run it. If it cannot run, follow the tutorial below and try to install the Conda environment and run it using the source code.

**Download**：

- Windows executable (might be slow when initial start): <a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/2.0.0/vse.exe">vse.exe</a> 

- Windows GPU version：<a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/2.0.0/vse_windows_gpu_v2.0.0.7z">vse_windows_gpu_v2.0.0.7z</a>

- Windows CPU version：<a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/2.0.0/vse_windows_cpu_v2.0.0.zip">vse_windows_cpu_v2.0.0.zip</a>

- MacOS：<a href="https://github.com/YaoFANGUK/video-subtitle-extractor/releases/download/0.1.0/vse_macOS_CPU.dmg">vse_macOS_CPU.dmg</a>


> **Provide your suggestions to improve this project in ISSUES & DISCUSSION**

**Pre-built Package Comparison**:

| Pre-built Package Name          | Python | Paddle | Environment                       | Supported Compute Capability Range |
|----------------------------------|--------|--------|-----------------------------------|------------------------------------|
| `vse-windows-cpu.7z`             | 3.12   | 3.0rc1 | No GPU, CPU only                  | Universal                         |
| `vse-windows-directml.7z`        | 3.12   | 3.0rc1 | Windows without Nvidia GPU         | Universal                         |
| `vse-windows-nvidia-cuda-10.2.7z`| 3.11   | 2.5.2  | CUDA 10.2                         | 3.0 – 7.5                          |
| `vse-windows-nvidia-cuda-11.8.7z`| 3.12   | 3.0rc1 | CUDA 11.8                         | 3.5 – 8.9                          |
| `vse-windows-nvidia-cuda-12.3.7z`| 3.12   | 3.0rc1 | CUDA 12.3                         | 5.0 – 9.0                          |

> NVIDIA provides a list of supported compute capabilities for each GPU model. You can refer to the following link: [CUDA GPUs](https://developer.nvidia.com/cuda-gpus) to check which CUDA version is compatible with your GPU.

**Recognition Mode Selection**:

| Mode Name     | GPU  | OCR Model Size | Subtitle Detection Engine | Notes            |
|---------------|------|-----------------|---------------------------|------------------|
| Fast          | Yes/No | Mini            | VideoSubFinder            |                  |
| Auto          | Yes    | Large           | VideoSubFinder            | Recommended      |
| Auto          | No     | Mini            | VideoSubFinder            | Recommended      |
| Precise       | Yes/No | Large           | VSE                       | Very slow        |

> The subtitle detection engine for both Windows/Linux environments is VideoSubFinder.

## Demo

- Graphic User Interface (GUI):

<p style="text-align:center;"><img src="https://github.com/YaoFANGUK/video-subtitle-extractor/raw/main/design/demo.gif" alt="demo.gif"/></p>


- Command Line Interface (CLI): 

[![Demo Video](https://s1.ax1x.com/2020/10/05/0JWVeJ.png)](https://www.bilibili.com/video/BV1t5411h78J "Demo Video")


## Running Online

- **Google Colab Notebook with free GPU**: <a href="https://colab.research.google.com/github/YaoFANGUK/video-subtitle-extractor/blob/main/google_colab_en.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

> PS: can only run CLI version on Google Colab


## Getting Started with Source Code 

#### 1. Install Python

Please ensure that you have installed Python 3.12+.

- Windows users can go to the [Python official website](https://www.python.org/downloads/windows/) to download and install Python.
- MacOS users can install using Homebrew:
  ```shell
  brew install python@3.12
  ```
- Linux users can install via the package manager, such as on Ubuntu/Debian:
  ```shell
  sudo apt update && sudo apt install python3.12 python3.12-venv python3.12-dev
  ```

#### 2. Install Dependencies

It is recommended to use a virtual environment to manage project dependencies to avoid conflicts with the system environment.

(1) Create and activate the virtual environment:
```shell
python -m venv videoEnv
```

- Windows:
```shell
videoEnv\\Scripts\\activate
```
- MacOS/Linux:
```shell
source videoEnv/bin/activate
```

#### 3. Create and Activate Project Directory

Change to the directory where your source code is located:
```shell
cd <source_code_directory>
```
> For example, if your source code is in the `tools` folder on the D drive and the folder name is `video-subtitle-extractor`, use:
> ```shell
> cd D:/tools/video-subtitle-extractor-main
> ```

#### 4. Install the Appropriate Runtime Environment

This project supports four runtime modes: CUDA (NVIDIA GPU acceleration), CPU (no GPU), DirectML (AMD, Intel, and other GPUs/APUs), and ONNX.

##### (1) CUDA (For NVIDIA GPU users)

> Make sure your NVIDIA GPU driver supports the selected CUDA version.

- Recommended CUDA 11.8, corresponding to cuDNN 8.6.0.

- Install CUDA:
  - Windows: [Download CUDA 11.8](https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe)
  - Linux:
    ```shell
    wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sudo sh cuda_11.8.0_520.61.05_linux.run
    ```
  - CUDA is not supported on MacOS.

- Install cuDNN (CUDA 11.8 corresponds to cuDNN 8.6.0):
  - [Windows cuDNN 8.6.0 Download](https://developer.download.nvidia.cn/compute/redist/cudnn/v8.6.0/local_installers/11.8/cudnn-windows-x86_64-8.6.0.163_cuda11-archive.zip)
  - [Linux cuDNN 8.6.0 Download](https://developer.download.nvidia.cn/compute/redist/cudnn/v8.6.0/local_installers/11.8/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz)
  - Follow the installation guide in the NVIDIA official documentation.

- Install PaddlePaddle GPU version (CUDA 11.8):
  ```shell
  pip install paddlepaddle-gpu==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
  pip install -r requirements.txt
  ```

##### (2) DirectML (For AMD, Intel, and other GPU/APU users)

- Suitable for Windows devices with AMD/NVIDIA/Intel GPUs.
- Install ONNX Runtime DirectML version:
  ```shell
  pip install paddlepaddle==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
  pip install -r requirements.txt
  pip install -r requirements_directml.txt
  ```

##### (3) ONNX (For macOS, AMD ROCm, and other GPU acceleration environments, not tested!)

- If using this method, DO NOT REPORT ISSUES.
- Suitable for Linux or macOS devices with AMD/Metal GPUs/Apple Silicon GPUs.
- Install ONNX Runtime DirectML version:
  ```shell
  pip install paddlepaddle==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
  pip install -r requirements.txt

  # Read documentation https://onnxruntime.ai/docs/execution-providers/
  # Choose the appropriate execution backend based on your device, modify the dependencies in requirements_directml.txt accordingly.

  # Example:
  # requirements_coreml.txt
  #   paddle2onnx==1.3.1
  #   onnxruntime-gpu==1.20.1
  #   onnxruntime-coreml==1.13.1

  pip install -r requirements_coreml.txt
  ```

##### (4) CPU Only (For systems without GPU or those not wanting to use GPU acceleration)

- Suitable for systems without GPU or those that do not wish to use GPU.
- Install the CPU version of PaddlePaddle:
  ```shell
  pip install paddlepaddle==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
  pip install -r requirements.txt
  ```

#### 5. Run the Program

- Run the graphical user interface version (GUI):
```shell
python gui.py
```

- Run the command-line interface version (CLI):
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

[![Powered by DartNode](https://dartnode.com/branding/DN-Open-Source-sm.png)](https://dartnode.com "Powered by DartNode - Free VPS for Open Source")


