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
