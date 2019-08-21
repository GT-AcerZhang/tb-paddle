# IMAGES

## Class SummaryWriter 的成员函数 add_image

函数定义：

```python
def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW')
    """Add image data to summary.

    Note that this requires the `pillow` package.
    
    :param tag: Data identifier.
    :type tag: string
    :param img_tensor: An `uint8` or `float` Tensor of shape `[channel, height, width]` where
                       `channel` is 1, 3, or 4. The elements in img_tensor can either have values
                       in [0, 1] (float32) or [0, 255] (uint8).
                       Users are responsible to scale the data in the correct range/type.
    :type img_tensor: numpy.array
    :param global_step: Global step value to record.
    :type global_step: int
    :param walltime: Optional override default walltime (time.time()) of event.
    :type walltime: float
    :param dataformats: 默认值为 'CHW'，各个字母的含义为 C-Channels, H-Height， W-Width。
                        可选值为 'HW','WH'，'CHW','CWH','HWC','WHC'。
    :type dataformats: string

    :Shape:
      If img_tensor.shape is `(3, H, W)`, corresponding dataformats should be 'CHW';
      If img_tensor.shape is `(H, W, 3)`, dataformats should be 'HWC';
      If image_tensor.shape is `(H, W)`, dataformats should be 'HW'.
    """
```

Demo-1 add_image-demo.py

```python
# coding=utf-8
import numpy as np
from PIL import Image
from tb_paddle import SummaryWriter


def random_crop(img):
    image = Image.open(img)
    w, h = image.size
    random_w = np.random.randint(0, w-100)
    random_h = np.random.randint(0, h-100)
    return image.crop((random_w, random_h, random_w + 100, random_h + 100))


# 创建 writer
writer = SummaryWriter('./log')

# 添加数据
step_num = 20
for step in range(step_num):
    image_path = "dog.jpg"
    # tag 为 image_1
    image_data_1 = np.array(random_crop(image_path))
    writer.add_image('image_1', image_data_1, step, dataformats='HWC')
    # tag 为 image_2
    image_data_2 = np.array(random_crop(image_path))
    writer.add_image('image_2', image_data_2, step, dataformats='HWC')

writer.close()
```

执行以下指令，启动服务器：

```
rm -rf ./log
python add_image-demo.py
tensorboard --logdir ./log/ --host 0.0.0.0 --port 6066
```

打开浏览器地址 [http://0.0.0.0:6066/](http://0.0.0.0:6066/) ，则可在 tensorboard 的 **IMAGES** 栏目查看图片：

<p align="center">
<img src="../screenshots/add_image.jpeg" width=600><br/>
图1. add_image - 在单个子框中显示一张图片 <br/>
</p>

##  Class SummaryWriter 的成员函数 add_images

函数定义：

```python
def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'):
    """Add batched (4D) image data to summary.
    Besides passing 4D (NCHW) tensor, you can also pass a list of tensors of the same size.
    In this case, the ``dataformats`` should be `CHW` or `HWC`.
    Note that this requires the ``pillow`` package.

    :param tag: Data identifier.
    :type tag: string
    :param img_tensor: Image data. The elements in img_tensor can either have
                 values in [0, 1] (float32) or [0, 255] (uint8).
                 Users are responsible to scale the data in the correct range/type.
    :type img_tensor: numpy.array
    :param global_step: Global step value to record.
    :type global_step: int
    :param walltime: Optional override default walltime (time.time()) of event.
    :type walltime: float

    :Shape:
      img_tensor: Default is :math:`(N, 3, H, W)`, If ``dataformats`` is specified,
                  other shape will be accepted. e.g. NCHW or NHWC.
    """
```

Demo-2 add_images-demo.py

```python
# coding=utf-8
from tb_paddle import SummaryWriter
import numpy as np

batch_num = 16
Channels = 3 
Height = 224 
Width = 180 
img_batch = np.zeros((batch_num, Channels, Height, Width))

H_W = Height * Width
for i in range(batch_num):
    img_batch[i, 0] = np.arange(0, H_W).reshape(Height, Width) / H_W / batch_num * i 
    img_batch[i, 1] = (1 - np.arange(0, H_W).reshape(Height, Width) / H_W) / batch_num * i 

writer = SummaryWriter('./log')
writer.add_images('image_batch', img_batch, 0, dataformats='NCWH')
writer.close()
```

运行程序`add_images-demo.py`，则可在 tensorboard 的 **IMAGES** 栏目查看图片。

<p align="center">
<img src="../screenshots/add_images.jpeg" width=600><br/>
图2. add_images - 在单个子框中显示多张图片 <br/>
</p>

## Class SummaryWriter 的成员函数 add_image_with_boxes

```python
def add_image_with_boxes(self, tag, img_tensor, box_tensor, global_step=None, 
                         walltime=None, dataformats='CHW', labels=labels, **kwargs):
    """Add image and draw bounding boxes on the image.
    :param tag: Data identifier.
    :type tag: string
    :param img_tensor: Image data.
    :type img_tensor: numpy.array
    :param box_tensor: Box data (for detected objects),
                       box should be represented as [x1, y1, x2, y2].
    :type box_tensor: numpy.array
    :param global_step: Global step value to record.
    :type global_step: int
    :param walltime: Optional override default walltime (time.time()) of event.
    :type walltime: float
    :param labels: The strings to be shown on each bounding box.
    :type labels: list of string

    :Shape:
        img_tensor: Default is :math:`(3, H, W)`.
                    It can be specified with `dataformat` agrument, e.g. CHW or HWC.

        box_tensor: NX4,  where N is the number of boxes and
                    each 4 elememts in a row represents (xmin, ymin, xmax, ymax).
    """
```

Demo-3 add_images_with_boxes-demo.py

```python
# coding=utf-8
import numpy as np
from PIL import Image
from tb_paddle import SummaryWriter


def random_crop(img):
    image = Image.open(img)
    w, h = image.size
    random_w = np.random.randint(0, w-100)
    random_h = np.random.randint(0, h-100)
    return image.crop((random_w, random_h, random_w + 100, random_h + 100))


# 创建 writer
writer = SummaryWriter('./log')

# 设置 box_tensoor 的取值
boxes_num = 2 
box_array = np.zeros([boxes_num,4])
box_array[0,0] = box_array[0,1] = 0 
box_array[0,2] = box_array[0,3] = 30
box_array[1,0] = box_array[1,1] = 50
box_array[1,2] = box_array[1,3] = 80

box_labels = ['face', 'mouth']

# 添加数据
step_num = 20
for step in range(step_num):
    image_path = "dog.jpg"
    image_data = np.array(random_crop(image_path))
    writer.add_image_with_boxes('image_with_boxes', image_data,
           box_array, step, dataformats='HWC', labels=box_labels)

writer.close()
```

运行程序`add_image_with_boxes.py`，则可在 tensorboard 的 **IMAGES** 栏目查看图片。

<p align="center">
<img src="../screenshots/add_images_with_boxes_labels.png" width=600><br/>
图3. add_image_with_boxes - 在单个子框中显示一张图片，并加上识别网格 <br/>
</p>

## class SummaryWriter 的成员函数 add_figure

```python
def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
    """Render matplotlib figure into an image and add it to summary.
    Note that this requires the ``matplotlib`` package.

    :param tag: Data identifier.
    :type tag: string
    :param figure: Figure or a list of figures.
    :type figure: matplotlib.pyplot.figure or list of matplotlib.pyplot.figure
    :param global_step: Global step value to record.
    :type global_step: int
    :param close: Flag to automatically close the figure.
    :type close: bool
    :param walltime: Optional override default walltime (time.time()) of event.
    :type walltime: float
    """
```

Demo-4 add_figure-demo.py

```python
# coding=utf-8
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt 

fig = plt.figure()

c1 = plt.Circle((0.2, 0.5), 0.2, color='r')
c2 = plt.Circle((0.8, 0.5), 0.2, color='r')

ax = plt.gca()
ax.add_patch(c1)
ax.add_patch(c2)
plt.axis('scaled')

from tb_paddle import SummaryWriter
writer = SummaryWriter('log')
writer.add_figure('matplotlib', fig)

writer.close()
```

运行程序`add_figure-demo.py`，则可在 tensorboard 的 **IMAGES** 栏目查看图片。

<p align="center">
<img src="../screenshots/add_figure.png" width=400><br/>
图4. add_figure - 显示 matplotlib 的画图 <br/>
</p>