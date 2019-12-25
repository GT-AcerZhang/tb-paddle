# 1 Discrete

## 创建 Discrete 对象

[class Discrete](../../tb_paddle/hparams_summary.py) 的初始化函数为：

```python
class Discrete(Domain):
    """离散值"""

    def __init__(self, values, dtype=None):
        """
        :param values: 可选的值列表，所有元素的数据类型都必须相同
        :param dtype: 只能为 `int`, `float`, `bool`, or `str` 中的一种，
            dtype 可为 None，此时会自动推断数据类型。
        """
```

demo-1 创建 Discrete 对象 

```python
>>> from tb_paddle import hparams_api as hp
>>> hp.Discrete([16, 32, 48])
Discrete([16, 32, 48])
>>> hp.Discrete([0.1, 0.2, 0.3])
Discrete([0.1, 0.2, 0.3])
>>> hp.Discrete([0.1, 2])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/Users/linshuliang/software/anaconda3/envs/paddle_py36/lib/python3.6/site-packages/tb_paddle/hparams_summary.py", line 103, in __init__
    % (value, self._dtype.__name__)
TypeError: dtype mismatch: not isinstance(2, float)
>>> hp.Discrete(['True', 'False'])
Discrete(['False', 'True'])
>>> optimizer = hp.Discrete(['Adam', 'SGD', 'Momentum', 'RMSPropOptimizer'])
>>> optimizer
Discrete(['Adam', 'Momentum', 'RMSPropOptimizer', 'SGD'])
```

## Discrete 对象的属性和成员函数

demo-2 Discrete 对象的成员函数和属性

```python
# coding=utf-8
from tb_paddle import hparams_api as hp

dropout_rate = hp.Discrete([0.1, 0.2, 0.3])
optimizer = hp.Discrete(['Adam', 'SGD', 'Momentum', 'RMSPropOptimizer'])

# 调用 __str__() 成员函数
print("dropout_rate:", dropout_rate)
print("optimizer:", optimizer)

# 调用 __repr__() 成员函数
print("repr(dropout_rate):", repr(dropout_rate))
print("repr(optimizer):", repr(optimizer))

# Discrete 对象有属性值 dtype
print("dropout_rate.dtype:", dropout_rate.dtype)
print("optimizer.dtype:", optimizer.dtype)

# Discrete 对象有属性值 values
print("dropout_rate.values:", dropout_rate.values)
print("optimizer.values:", optimizer.values)

# 成员函数 sample_uniform 随机返回一个值
print("dropout_rate.sample_uniform():", dropout_rate.sample_uniform())
print("optimizer.sample_uniform():", optimizer.sample_uniform())
```