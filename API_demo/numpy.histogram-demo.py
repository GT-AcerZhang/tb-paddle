# coding=utf-8
import numpy as np
"""
def numpy.histogram(a, bins=10, range=None, weights=None, density=None)
    '''
    将 a 的各个数值按 bins 分类。
        
    :param:
      > a     : Input data.
                DataType : flattend array
      > bins  : 直方图的柱子.
                DataType : int or sequence of scalars.
                if bins is an integer, it defines the number of equal width bins in the given range.
                It defines a monotonically increasing array of bin edges, 
                including the rightmost edge, allowing for non-uniform bin widths. 
      > range : 数据范围.
                DataType : (float, float)
                The lower and upper range of bins.
                If not provided, range is simply (a.min(), a.max())
      > density: DataType bool.
                If False, the result will contain the number of samples in each bin.
                If True, the result is the value of probability density function at the bin.

    :return:
      > hist      : DataType array, The values of histogram.
      > bin_edges : DataType array, the bin edges.

    :Tips:
      len(bin_edges) = len(hist) + 1  
    '''
"""

print(np.histogram([1, 2, 1], bins=[0, 1, 2, 3]))
# (array([0, 2, 1]), array([0, 1, 2, 3]))

print(np.histogram(np.arange(4), bins=np.arange(5), density=True))
# (array([ 0.25,  0.25,  0.25,  0.25]), array([0, 1, 2, 3, 4]))

print(np.histogram([[1, 2, 1], [1, 0, 1]], bins=[0,1,2,3]))
# (array([1, 4, 1]), array([0, 1, 2, 3]))
