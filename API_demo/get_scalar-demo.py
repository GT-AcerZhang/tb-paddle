# coding=utf-8
import os
import time
import numpy

def log_scalar(dir_path, tag):
    """log scalar in directory os.path.join(dir_path, tag)
    
    :param tag: tag of scalar
    :type tag: str
    :return: None
    """
    from tb_paddle import SummaryWriter

    writer = SummaryWriter(logdir=os.path.join(dir_path, tag))

    upper_bound = numpy.random.randint(low=5, high=20)
    for i in range(1, upper_bound):
        writer.add_scalar(tag, numpy.random.randint(100), i)

    writer.close()


if __name__ == '__main__':
    from tb_paddle import SummaryReader
    parse_dir = "log"
    reader = SummaryReader(parse_dir=parse_dir)
    
    tag1 = "first"
    log_scalar(parse_dir, tag1)
    print("scalar: {}".format(tag1))
    res1 = reader.get_scalar(tag1)
    print(res1) 

    tag2 = "second"
    print("scalar: {}".format(tag2))
    log_scalar(parse_dir, tag2)
    res2 = reader.get_scalar(tag2)
    print(res2)
