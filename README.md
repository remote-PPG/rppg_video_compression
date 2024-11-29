# rPPG Video Compression

> The project only publishes the project framework and models, and the algorithms for individual research reproduction are not uploaded and published.

## 安装 ffmpeg
[FFmpeg (https://ffmpeg.org)](https://ffmpeg.org/)

## 环境

### 克隆项目后克隆子模块
```
git submodule update --init --recursive
```

### 按照文件夹 *rppg_toolbox* 中的 *readme.md* 步骤搭建环境

```
# conda deactivate
# conda env remove -n rppg_video_compression -y

conda create -n rppg_video_compression python=3.9.19 pip=24.0 -y

conda activate rppg_video_compression

python -m pip install jupyter notebook -U

```

### 本项目的依赖
```
pip install -r requirements.txt
```

### 更新子项目 *rppg_toolbox*
```
git submodule update --remote
```

### 模型训练, 会在当前目录```./out/```下保存模型
```
# 修改PhysNet_Train.py 数据集目录等配置，并执行
python PhysNet_Train.py
```