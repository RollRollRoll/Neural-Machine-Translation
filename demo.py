#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
demo.py: A demo about processing variable length sequence input
"""

import torch
from torch import nn
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def rnn_forwarder(rnn, inputs, seq_lengths):
    """
    :param rnn: RNN instance
    :param inputs: FloatTensor, shape [batch, time, dim] if rnn.batch_first else [time, batch, dim]
    :param seq_lengths: LongTensor shape [batch]
    :return: the result of rnn layer,
    """
    batch_first = rnn.batch_first
    # assume seq_lengths = [3, 5, 2]
    # 对序列长度进行排序(降序), sorted_seq_lengths = [5, 3, 2]
    # indices 为 [1, 0, 2], indices 的值可以这么用语言表述
    # 原来 batch 中在 0 位置的值, 现在在位置 1 上.
    # 原来 batch 中在 1 位置的值, 现在在位置 0 上.
    # 原来 batch 中在 2 位置的值, 现在在位置 2 上.
    sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)

    # 如果我们想要将计算的结果恢复排序前的顺序的话,
    # 只需要对 indices 再次排序(升序),会得到 [0, 1, 2],
    # desorted_indices 的结果就是 [1, 0, 2]
    # 使用 desorted_indices 对计算结果进行索引就可以了.
    _, desorted_indices = torch.sort(indices, descending=False)

    # 对原始序列进行排序
    if batch_first:
        inputs = inputs[indices]
    else:
        inputs = inputs[:, indices]

    packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs,
                                                      sorted_seq_lengths.cpu().numpy(),
                                                      batch_first=batch_first)
    res, state = rnn(packed_inputs)
    padded_res, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=batch_first)

    # 恢复排序前的样本顺序
    if batch_first:
        desorted_res = padded_res[desorted_indices]
    else:
        desorted_res = padded_res[:, desorted_indices]
    return desorted_res


if __name__ == "__main__":
    bs = 3
    max_time_step = 6
    feat_size = 7
    hidden_size = 12
    seq_lengths = torch.tensor([3, 5, 2]).long()

    rnn = nn.GRU(input_size=feat_size,
                 hidden_size=hidden_size, batch_first=True, bidirectional=False)
    rnn.to(device)

    x = torch.tensor(np.random.randn(bs, max_time_step, feat_size), dtype=torch.float32)
    # x = torch.tensor(torch.FloatTensor(bs, max_time_step, feat_size).clone().detach())
    x_gpu = x.to(device)
    print("x:\ndevice: {} is_cuda: {} id: {} x.shape : {}".format(x.device, x.is_cuda, id(x), x.shape))
    print("x_gpu:\ndevice: {} is_cuda: {} id: {} x_gpu.shape : {}".format(x_gpu.device, x_gpu.is_cuda, id(x_gpu),
                                                                          x_gpu.shape))

    using_packed_res = rnn_forwarder(rnn, x_gpu, seq_lengths)
    print("using_packed_res :\n {}".format(using_packed_res))
    # 不使用 pack_paded, 用来和上面的结果对比一下.
    not_packed_res, _ = rnn(x_gpu)
    print("not_packed_res :\n {}".format(not_packed_res))
