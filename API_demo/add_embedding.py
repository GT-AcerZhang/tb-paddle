# coding = utf-8
import numpy as np
import paddle
from tb_paddle import SummaryWriter

writer = SummaryWriter('./log')

BATCH_SIZE = 1024
reader_shuffle = paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=5120)

train_reader = paddle.batch(
    reader_shuffle,
    batch_size=BATCH_SIZE)

mat = np.zeros([BATCH_SIZE, 784])
metadata = np.zeros(BATCH_SIZE)
for step_id, data in enumerate(train_reader()):
    # type(data) : <class 'list'>
    # len(data)  : BATCH_SIZE
    for i in range(len(data)):
        # type(data[i][0]) : <class 'numpy.ndarray'>
        # data[i][0].shape : (784,)
        mat[i] = data[i][0]
        # type(data[i][1]) : int
        metadata[i] = data[i][1]

    label_img = mat.reshape(BATCH_SIZE, 1, 28, 28)
    print('Step %d' % step_id)

writer.add_embedding(mat=mat, metadata=metadata,
    label_img=label_img, global_step=step_id)

# 如果只有 add_embedding, 没有其他数据添加语句，前端会找不到 embedding 的数据
writer.add_scalar('echo', 1, 0)

writer.close()