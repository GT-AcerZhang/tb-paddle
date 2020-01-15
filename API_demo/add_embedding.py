# coding=utf-8
import numpy as np
import paddle
from tb_paddle import SummaryWriter

writer = SummaryWriter('./log')
BATCH_SIZE = 1024
train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=5120),
    batch_size=BATCH_SIZE)
mat = np.zeros([BATCH_SIZE, 784])
metadata = np.zeros(BATCH_SIZE)
embedding_data = {}
for step_id, data in enumerate(train_reader()):
    # type(data) : <class 'list'>
    # len(data)  : BATCH_SIZE
    # type(data[i]) : <class 'tuple'>
    # type(data[i][0]) : <class 'numpy.ndarray'>
    # data[i][0].shape : (784,)
    # type(data[i][1]) : <class 'int'>
    embedding_data = data
    if step_id > 0:
        break 

for i in range(len(embedding_data)):
    mat[i] = embedding_data[i][0]
    metadata[i] = embedding_data[i][1]

label_img = mat.reshape(BATCH_SIZE, 1, 28, 28)
writer.add_embedding(
    mat=mat,
    metadata=metadata,
    label_img=label_img,
    global_step=step_id)

writer.add_scalar('echo', 1, 0)
writer.close()
