# MESH

TensorBoard 的**MESH**栏目显示网格和点云。

class SummaryWriter 中用于打点 3D 数据的成员函数为`add_mesh`。

函数 `add_mesh` 的定义与实现均在文件[../tb_paddle/summary_writer.py](../tb_paddle/summary_writer.py) 中。

## Class SummaryWriter 的成员函数 add_mesh

网格和点云(Meshes and points cloud)是表示3D 图形的重要数据类型，目前已广泛用于计算机视觉和计算机图形学中。随着 3D 数据在 VR/AR 等领域的普及，研究人员面临着许多新的挑战，例如从 2D 数据来实现 3D 几何重构、3D 云语义分割，等等。因此，可视化 3D 形状的计算结果，有助于改进模型和分析算法的有效性。目前， [Tensorflow graphics](https://github.com/tensorflow/graphics) 通过 Tensoboard 的 **MESH** 栏目，提供了可视化 3D 形状的功能。

Demo-1 add_mesh-demo.py

```python
# coding=utf-8
import numpy as np
from tb_paddle import SummaryWriter

vertices_tensor = np.array([[
    [1, 1, 1],
    [-1, -1, 1],
    [1, -1, -1],
    [-1, 1, -1],
    [-1,-1,-1],
    [1, 1, -1]
]], dtype=float)

colors_tensor = np.array([[
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [152, 0, 255],
    [180, 212, 0],
    [0, 99, 65]
]], dtype=int)

faces_tensor = np.array([[
    [0, 2, 3],
    [0, 3, 1],
    [0, 1, 2],
    [1, 3, 2],
    [4, 1, 3],
    [1, 4, 5],
]], dtype=int)

writer = SummaryWriter('log')
writer.add_mesh('my_mesh', vertices=vertices_tensor, 
    colors=colors_tensor, faces=faces_tensor)

writer.close()
```

执行以下指令，启动服务器

```
python add_mesh-demo.py
tensorboard --logdir ./log/ --host 0.0.0.0 --port 6066
```

打开浏览器地址 [http://0.0.0.0:6066/](http://0.0.0.0:6066/) ，
则可在 tensorboard 的 **MESH** 栏目中查看3D图形：

<p align="center">
<img src="../screenshots/add_mesh.png", width=600><br/>
图1. add_mesh - 显示 3D 点云或网格
</p>
