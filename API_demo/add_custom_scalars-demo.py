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
