# tb-paddle
[![Build Status](https://travis-ci.org/linshuliang/tb-paddle.svg?branch=master)](https://travis-ci.org/linshuliang/tb-paddle)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://github.com/linshuliang/tb-paddle/blob/master/README.md)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 简介

tb-paddle 是一个用于在 TensorBoard 中查看 Paddle 打点数据的可视化工具。

目前 tb-paddle 支持 SCALARS, HISTOGRAMS, DISTRIBUTIONS, GRAPHS, IMAGES, TEXT,
AUDIO, PROJECTOR,PR CURVES, MESH, CUSTOM SCALARS 这11个栏目的功能。

|栏目|展示图表|作用|
|:----:|:---:|:---|
|[SCALARS](instructions/SCALARS_instructions.md)|折线图|显示损失函数值、准确率等标量数据|
|[HISTOGRAMS, DISTRIBUTIONS](instructions/HISTOGRAMS_DISTRIBUTIONS_instructions.md)|分布图|显示行向量数据的数值分布与变化趋势，便于查看权重矩阵、偏置项、梯度等参数的变化|
|[GRAPHS](instructions/GRAPHS_instructions.md)|计算图|显示神经网络的模型结构|
|[IMAGES](instructions/IMAGES_instructions.md)|图片和视频|显示图片和视频|
|[AUDIO](instructions/AUDIO_instructions.md)|音频|播放音频|
|[TEXT](instructions/TEXT_instructions.md)|文本|显示文本|
|[PROJECTOR](instructions/PROJECTOR_instructions.md)|交互式的嵌入可视化|通过降维方法将高维数据嵌入到 2D/3D 中显示，支持 PCA, t-SNE, UMAP, CUSTOM 这四种降维方法|
|[PR CURVES](instructions/PR-CURVES_instructions.md)|Precision-Recall曲线|根据预测的概率值及其对应的准确答案计算[Precision-Recall](https://en.wikipedia.org/wiki/Precision_and_recall) 曲线|
|[MESH](instructions/MESH_instructions.md)|网格和点云|显示3D图形的网格和点云(Meshes and points cloud)|
|[CUSTOM SCALARS](instructions/CUSTOM_SCALARS_instructions.md)|组合折线图|显示用户自定义组合的折线图|

## 特别致谢

tb-paddle 是在 [tensorboardX](https://github.com/lanpa/tensorboardX) 的基础上修改的，
tb-paddle 的框架和 API 接口均沿用了 tensorboardX。与 tensorboardX 不同的是，
tb-paddle 的API接口的参数类型为`numpy.ndarray`和Python基本数据类型，
并根据 Paddle 框架重新实现了GRAPHS栏目的计算图显示。
此处由衷感谢[Tzu-Wei Huang](https://github.com/lanpa)的开源贡献。

## 安装

```
# 安装 tb-nightly
pip install tb-nightly==1.15.0a20190818

# 源码安装 tb-paddle
git clone https://github.com/linshuliang/tb-paddle.git && cd tb-paddle && python setup.py install
```

## 创建 SummaryWriter 类的对象

使用 tb-paddle，首先得创建类`SummaryWriter`的对象，然后才能调用对象的成员函数来添加打点数据。

创建 [class SummaryWriter](tb_paddle/writer.py#L177) 的初始化函数的定义：

```python
def __init__(self,
            logdir=None,
            flush_secs=10,
            max_queue=1000,
            purge_step=None,
            comment='',
            filename_suffix='',
            write_to_disk=True,
            **kwargs):
```

其中各个参数的含义为：

* `logdir` ：指定日志文件的存放路径，如果指定路径中没有 tfevents 文件，则新建一个 tfevents 文件，否则会向已有的 tfevents 文件中写数据。`logdir`的实参可以为`None`，则存放路径设为`./runs/DATETIME_HOSTNAME/`；
* `flush_secs` ：将打点数据从缓冲写到磁盘中，单位为秒；
* `max_queue` ： 缓冲区队列的最大长度；
* `purge_step` ：截断步数，重启打点时，tfevents 文件中步数大于`purge_step`的数据将被清除；
* `comment` ：如果`logdir`为`None`，则在默认存放路径中添加后缀。如果`logdir`不是`None`，那么该参数没有任何作用；
* `filename_suffix` ：event 文件名后缀；
* `write_to_disk` ：是否将打点数据写到磁盘。

## TensorBoard 启动命令

启动 TensorBoard 的命令为`tensorboard`，输入 `tensorboard --helpful` 则可查看此命令的帮助文档。

通常会用到`--logdir`, `--host`, `--port`, `--reload_interval`这几个选项：

```
tensorboard --logdir <path/to/dir> --host <host_name> --port <port_num> --reload_interval <time_secs>
```

这几个选项的详细解释:

1. `--logdir`

`--logdir` 用于指定 `tfevents` 文件的存放路径，可以同时指定多个目录，比如：

```
tensorboard --logdir ExperimentA:path/to/A_dir,ExperimentB:another/path/to/B_dir
```

只需在不同目录名间加上逗号(`,`) ，则可同时指定多个目录。

事实上，TensorBoard 会自动检查指定目录下的所有子目录中的 `tfevents` 文件，并在前端网页中
按 `Runs` 分类，比如目录结构为：

```
log
|
|____log_mnist
|    |
|    |___logtest
|    |
|    |___logtrain
|   
|____paddle_log
```

则在 TensorBoard 前端页面的左侧栏中显示为：

<p align="center">
<img src="./screenshots/tensorboard_manuals/Runs.png" width=300><br/>
图1. TensorBoard Runs 选项 - 按目录分类 <br/>

2. `--host`

在本机运行`tensorboard`命令， `--host` 指定为 `0.0.0.0`， 在服务器上运行，`--host` 指定为服务器的地址。

3. `--port`

在本机运行`tensorboard`命令，`--port` 指定为 `6***`， 在服务器上运行，`--port` 指定为 `8***`。

4. `--reload_interval`

后端读取 `tfevents` 文件数据的时间间隔，单位为秒，默认为5秒。
