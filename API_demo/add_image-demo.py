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
    writer.add_image('crop/image [1]', image_data_1, step, dataformats='HWC')
    # tag 为 image_2
    image_data_2 = np.array(random_crop(image_path))
    writer.add_image('crop/image [2]', image_data_2, step, dataformats='HWC')

writer.close()
