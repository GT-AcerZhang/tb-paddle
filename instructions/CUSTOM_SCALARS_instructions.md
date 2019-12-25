# CUSTOM SCALARS

TensorBoard 的 **CUSTOM SCALARS** 栏目显示用户自定义组合的折线图。

通过函数`add_scalar`和`add_scalars`添加标量数据，会根据`tags`和`runs`来进行分类，进而在 Tensorboard 中显示。举个例子，用户打点了`resnet`模型和`vggnet`模型的`loss`值，这两个模型的`loss`值的折线图会根据`tags`和`runs`的值被分类到不同的位置，用户难以直接比较两个模型的`loss`值。

通过收集函数`add_scalar`的`tag`和`runs`，可组合出新的`layout`，
将`layout`作为实参传入 class SummarWriter 的成员函数：

* <a href="#1"> add_custom_scalars </a>
* <a href="#2"> add_custom_scalars_multilinechart </a>
* <a href="#3"> add_custom_scalars_marginchart </a>

就可以在同一个图中同时绘制多条折线，以直观地进行数据比较和分析。

以上 API 的定义与实现均在文件[../tb_paddle/summary_writer.py](../tb_paddle/summary_writer.py) 中。

<a name="1"></a>
## Class Summary 的成员函数 add_custom_scalars

Demo-1 add\_custom\_scalars-demo.py

```python
# coding=utf-8
from numpy.random import randn, rand
from tb_paddle import SummaryWriter

with SummaryWriter('log') as writer:
    for n_iter in range(100):

        writer.add_scalar('SH/0001SH', 3000 + 240*randn(), n_iter)
        writer.add_scalar('SH/600519', 1000 + 100*randn(), n_iter)

        t = randn()
        writer.add_scalar('nasdaq/microsoft', t, n_iter)
        writer.add_scalar('nasdaq/google', t - 1, n_iter)
        writer.add_scalar('nasdaq/cisco', t + 1, n_iter)
        writer.add_scalar('nasdaq/intel', t + 2, n_iter)

        writer.add_scalar('dow/aaa', rand(), n_iter)
        writer.add_scalar('dow/bbb', rand(), n_iter)
        writer.add_scalar('dow/ccc', rand(), n_iter)

    layout = {'China': {'SH': ['Multiline', ['SH/0001SH', 'SH/600519']]},
              'USA': {'dow': ['Margin', ['dow/aaa', 'dow/bbb', 'dow/ccc']],
                      'nasdaq': ['Multiline', ['nasdaq/microsoft', 'nasdaq/google', 'nasdaq/cisco', 'nasdaq/intel']]}}

    writer.add_custom_scalars(layout)
```

运行程序，启动服务器：

```
python add_custom_scalars-demo.py
tensorboard --logdir ./log/ --host 0.0.0.0 --port 6066
```

打开浏览器地址 [http://0.0.0.0:6066/](http://0.0.0.0:6066/) ，则可在tensorboard 的 CUSTOM SCALAR 栏目中查看下图：

<p align="center">
<img src="../screenshots/add_custom_scalars.png" width=1000><br/>
图1. add_custom_scalars - 显示折线图 <br/>
</p>

<a name="2"></a>
## Class Summary 的成员函数 add_custom_scalars_multilinechart

函数 `add_custom_scalars_multilinechart` 用于组合多条折线，
便于直观地查看，进而有助于数据比较和分析。

Demo-2 add_custom_scalars_multilinechart.py

```python
# coding=utf-8
from numpy.random import randn
from tb_paddle import SummaryWriter

with SummaryWriter('log') as writer:
    for n_iter in range(100):
        writer.add_scalar('SZ/000858', 123 + 12*randn(), n_iter)
        writer.add_scalar('SZ/000568', 81 + 8*randn(), n_iter)

    writer.add_custom_scalars_multilinechart(
        ['SZ/000858', 'SZ/000568'], category='China', title='SZ')
```

运行程序`add_custom_scalars_multilinechart`，则可在 tensorboard 的 CUSTOM SCALAR 栏目中查看下图：

<p align="center">
<img src="../screenshots/add_custom_scalars_multilinechart.png" width=400><br/>
图2. add_custom_scalars_multilinechart - 显示 `Multiline` 折线图 <br/>
</p>

<a name="3"></a>
## Class Summary 的成员函数 add_custom_scalars_marginchart

函数 `add_custom_scalars_marginchart` 用于可视化置信区间。

Demo-3 add_custom_scalars_marginchart.py

```python
# coding=utf-8
from numpy.random import randn
from tb_paddle import SummaryWriter
import math

with SummaryWriter('log') as writer:
    for n_iter in range(100):
        writer.add_scalar('HK/00700', 360 + math.sqrt(360)*randn(), n_iter)
        writer.add_scalar('HK/000568', 9.78  + math.sqrt(9.78)*randn(), n_iter)
        writer.add_scalar('HK/00001', 80 + math.sqrt(80)*randn(), n_iter)

    writer.add_custom_scalars_marginchart(
       ['HK/00700', 'HK/00001', 'HK/000568'], category='China', title='HK')
```

运行程序`add_custom_scalars_marginchart.py`，则可在 tensorboard 的 CUSTOM SCALAR 栏目中查看下图：

<p align="center">
<img src="../screenshots/add_custom_scalars_marginchart.png" width=400><br/>
图3. add_custom_scalars_marginchart - 显示 `Margin` 折线图 <br/>
</p>
