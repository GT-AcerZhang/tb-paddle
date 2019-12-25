# coding=utf-8
from tb_paddle import SummaryWriter

writer = SummaryWriter('./log')

for step in range(10):
    text = 'The text support the markdown format.  \nThis is line ' + str(step)  
    writer.add_text('LSTM', text, step)
    writer.add_text('rnn', text, step)

writer.close()
