"""
the modules that follows CNN feature extractor,
mainly RNN, but can also be attention
"""
from copy import deepcopy

from easydict import EasyDict as ED


__all__ = [
    "lstm",
    "attention",
]


lstm = ED()
lstm.bias = True
lstm.dropouts = 0.2
lstm.bidirectional = True
lstm.retseq = True
lstm.hidden_sizes = [12*16, 12*16]


attention = ED()
# almost the same with lstm, but the last layer is an attention layer
attention.head_num = 12
attention.bias = True
attention.dropouts = 0.2
attention.bidirectional = True
attention.hidden_sizes = [12*24, 12*6]
