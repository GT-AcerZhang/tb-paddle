# coding=utf-8
from tb_paddle import SummaryWriter

writer = SummaryWriter('./log')

for step in range(10):
    text = 'This is text ' + str(step)  
    writer.add_text('LSTM', text, step)
    writer.add_text('rnn', text, step)

writer.close()
