# VIDEO

使用 Tensorboard 的 **VIDEO** 功能，必须先安装`moviepy`：

```
pip install moviepy
```

## class SummaryWriter 的成员函数 add_video

函数定义：

```python
def add_video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None):
    """Add video data to summary.
    
    :param tag: Data identifier.
    :type tag: string
    :param vid_tensor: Video data.
    :type vid_tensor: numpy.array
    :param global_step: Global step value to record.
    :type global_step: int
    :param fps: Frames per second.
    :type fps: float or int
    :param walltime: Optional override default walltime (time.time()) of event.
    :type walltime: float

    :Shape:
        vid_tensor:  `(Picture_num, Frame_num, Channel, Height, Weight)`，其中：
                      Picture_num 表示每一桢包括多少张(C,H,W)的图片；
                      Frame_num 表示该数据总计多少帧；
                      若 vid_tensor 的元素的数据类型为`uint8`，则取值范围是 [0, 255]；
                      若 vid_tensor 的元素的数据类型为`float`，则取值范围是 [0,1]。
```

Demo-1 add_video-demo.py

```python
# coding=utf-8
import numpy as np
import paddle
from tb_paddle import SummaryWriter
import matplotlib
matplotlib.use('TkAgg')

writer = SummaryWriter('log')

BATCH_SIZE = 768
reader_shuffle = paddle.reader.shuffle(
                     paddle.dataset.mnist.train(), buf_size=5120)

train_reader = paddle.batch(reader_shuffle, batch_size=BATCH_SIZE)

mat = np.zeros([BATCH_SIZE, 784])
for step_id, data in enumerate(train_reader()):
    # type(data) : <class 'list'>
    # len(data)  : BATCH_SIZE
    for i in range(len(data)):
        # type(data[i][0]) : <class 'numpy.ndarray'>
        # data[i][0].shape : (784,)
        mat[i] = data[i][0]

video_data = mat.reshape((16, 48, 1, 28, 28))
writer.add_video('mnist_video_fps4', vid_tensor=video_data)
writer.add_video('mnist_video_fps1', vid_tensor=video_data, fps=1)

writer.close()
```

执行以下指令，启动服务器：

```
python add_video-demo.py
tensorboard --logdir ./log/ --host 0.0.0.0 --port 6066
```

打开浏览器地址 [http://0.0.0.0:6066/](http://0.0.0.0:6066/) ，则可在 tensorboard 的**IMAGES**栏目中查看视频：

<p align="center">
<img src="../screenshots/add_video.png" width=600><br/>
图1. add_video - 展示视频 <br/>
</p>
