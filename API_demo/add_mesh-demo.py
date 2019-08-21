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
