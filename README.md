## 0. 下载项目文件，将工作目录切换到项目文件所在目录
```shell
cd video-subtitle-extractor
```

## 1. 下载安装Anaconda
<a href="https://www.anaconda.com/products/individual">https://www.anaconda.com/products/individual</a>

## 2. 使用conda创建项目虚拟环境并激活环境
```shell
conda create --name videoEnv python=3.8
conda activate videoEnv  
```

## 3. 使用pip安装依赖文件
```shell
pip install -r requirements.txt
```

## 4. 编译bbox文件
```shell
cd utils/bbox
chmod +x make.sh
./make.sh
```

## 5. 运行程序
```shell
python3 ./main/demo.py
```

### 6.性能检测（可选）
```shell
python3 ./main/accuracyCal.py
```

Note:
1. install requirements before running
2. download trained model files from: (other Supplementary material also can be found here)
https://drive.google.com/drive/folders/1AuthnUK7bqYOlLcKi1GkoLyHbq3GyjfX?usp=sharing
3. unzip checkpoints_mlt.zip and put it into the checkpoints_mlt directory
4. demo video can be found at: https://youtu.be/0gq8FQHb448

