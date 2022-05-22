import os
import re
import math
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from utils import *


class RNNModel(nn.Module):
    """循环神经网络模型。"""

    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），`num_directions`应该是2，否则应该是1。
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将`Y`的形状改为(`时间步数`*`批量大小`, `隐藏单元数`)。
        # 它的输出形状是 (`时间步数`*`批量大小`, `词表大小`)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            # `nn.GRU` 以张量作为隐藏状态
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        else:
            # `nn.LSTM` 以张量作为隐藏状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


if __name__ == '__main__':
    data_dir = './data'
    use_random_iter = False
    batch_size, num_steps = 32, 35
    with open(os.path.join(data_dir, 'timemachine.txt'), 'r') as f:
        lines = f.readlines()
    tm_lines = [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]
    train_iter = SeqDataLoader(tm_lines, batch_size, num_steps, use_random_iter, max_tokens=10000)
    vocab = train_iter.vocab

    # 定义rnn层初始化隐藏状态
    vocab_size, num_hiddens = len(vocab), 256
    num_inputs = vocab_size
    rnn_layer = nn.LSTM(num_inputs, num_hiddens)

    # 基本参数
    num_epochs, lr = 500, 1
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())] \
        if torch.cuda.is_available() else [torch.device('cpu')]
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(devices[0])
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr)

    # train
    for epoch in range(num_epochs):
        state = None
        metric = Accumulator(2)  # Sum of training loss, no. of tokens
        data_iter = train_iter if (epoch + 1) % 5 != 0 else tqdm(train_iter, desc=f"Processing {epoch + 1:02}")
        for X, Y in data_iter:
            if state is None or use_random_iter:
                # Initialize `state` when either it is the first iteration or
                # using random sampling
                state = net.begin_state(batch_size=X.shape[0], device=devices[0])
            else:
                if isinstance(net, nn.Module) and not isinstance(state, tuple):
                    # `state` is a tensor for `nn.GRU`
                    state.detach_()
                else:
                    # `state` is a tuple of tensors for `nn.LSTM` and
                    # for our custom scratch implementation
                    for s in state:
                        s.detach_()
            y = Y.T.reshape(-1)
            X, y = X.to(devices[0]), y.to(devices[0])
            y_hat, state = net(X, state)
            l = loss(y_hat, y.long())
            optimizer.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            optimizer.step()
            metric.add(l * torch.numel(y), torch.numel(y))
        if (epoch + 1) % 5 == 0:
            print(f'perplexity {math.exp(metric[0] / metric[1]):.1f}')
    torch.save(net.state_dict(), 'models/lstm.pth')

    # predict
    model = net
    model.load_state_dict(torch.load('./models/lstm.pth', map_location=devices[0]))
    print(predict('time traveller', 50, model, vocab, devices[0]))
