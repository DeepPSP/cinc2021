"""
the most basic CNN
"""
from copy import deepcopy
from typing import Union, Optional, Tuple, Sequence, NoReturn
from numbers import Real, Number

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
import torch
from torch import nn
from torch import Tensor
from easydict import EasyDict as ED

from ...cfg import DEFAULTS
from ...utils.utils_nn import (
    compute_maxpool_output_shape,
    compute_module_size,
    SizeMixin,
)
from ...utils.misc import dict_to_str
from ...models._nets import (
    Conv_Bn_Activation,
    NonLocalBlock, SEBlock, GlobalContextBlock,
)


if DEFAULTS.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "VGGBlock", "VGG16",
]


class VGGBlock(SizeMixin, nn.Sequential):
    """ finished, checked,

    building blocks of the CNN feature extractor `VGG16`
    """
    __DEBUG__ = False
    __name__ = "VGGBlock"

    def __init__(self, num_convs:int, in_channels:int, out_channels:int, groups:int=1, **config) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        num_convs: int,
            number of convolutional layers of this block
        in_channels: int,
            number of channels in the input
        out_channels: int,
            number of channels produced by the convolutional layers
        groups: int, default 1,
            connection pattern (of channels) of the inputs and outputs
        config: dict,
            other parameters, including
            filter length (kernel size), activation choices,
            weight initializer, batch normalization choices, etc. for the convolutional layers;
            and pool size for the pooling layer
        """
        super().__init__()
        self.__num_convs = num_convs
        self.__in_channels = in_channels
        self.__out_channels = out_channels
        self.__groups = groups
        self.config = ED(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        self.add_module(
            "cba_1",
            Conv_Bn_Activation(
                in_channels, out_channels,
                kernel_size=self.config.filter_length,
                stride=self.config.subsample_length,
                groups=self.__groups,
                activation=self.config.activation,
                kw_activation=self.config.kw_activation,
                kernel_initializer=self.config.kernel_initializer,
                kw_initializer=self.config.kw_initializer,
                batch_norm=self.config.batch_norm,
            )
        )
        for idx in range(num_convs-1):
            self.add_module(
                f"cba_{idx+2}",
                Conv_Bn_Activation(
                    out_channels, out_channels,
                    kernel_size=self.config.filter_length,
                    stride=self.config.subsample_length,
                    groups=self.__groups,
                    activation=self.config.activation,
                    kw_activation=self.config.kw_activation,
                    kernel_initializer=self.config.kernel_initializer,
                    kw_initializer=self.config.kw_initializer,
                    batch_norm=self.config.batch_norm,
                )
            )
        self.add_module(
            "max_pool",
            nn.MaxPool1d(self.config.pool_size, self.config.pool_stride)
        )

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        output = super().forward(input)
        return output

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, None]]:
        """ finished, checked,

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shape: sequence,
            the output shape of this block, given `seq_len` and `batch_size`
        """
        num_layers = 0
        for module in self:
            if num_layers < self.__num_convs:
                output_shape = module.compute_output_shape(seq_len, batch_size)
                _, _, seq_len = output_shape
            else:
                output_shape = compute_maxpool_output_shape(
                    input_shape=[batch_size, self.__out_channels, seq_len],
                    kernel_size=self.config.pool_size,
                    stride=self.config.pool_size,
                    channel_last=False,
                )
            num_layers += 1
        return output_shape


class VGG16(SizeMixin, nn.Sequential):
    """ finished, checked,

    CNN feature extractor of the CRNN models proposed in refs of `ECG_CRNN`
    """
    __DEBUG__ = False
    __name__ = "VGG16"

    def __init__(self, in_channels:int, **config) -> NoReturn:
        """ finished, checked,
        
        Parameters
        ----------
        in_channels: int,
            number of channels in the input
        config: dict,
            other hyper-parameters of the Module, including
            number of convolutional layers, number of filters for each layer,
            and more for `VGGBlock`.
            key word arguments that have to be set:
            num_convs: sequence of int,
                number of convolutional layers for each `VGGBlock`
            num_filters: sequence of int,
                number of filters for each `VGGBlock`
            groups: int,
                connection pattern (of channels) of the inputs and outputs
            block: dict,
                other parameters that can be set for `VGGBlock`
            for a full list of configurable parameters, ref. corr. config file
        """
        super().__init__()
        self.__in_channels = in_channels
        # self.config = deepcopy(ECG_CRNN_CONFIG.cnn.vgg16)
        self.config = ED(deepcopy(config))
        if self.__DEBUG__:
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")

        module_in_channels = in_channels
        for idx, (nc, nf) in enumerate(zip(self.config.num_convs, self.config.num_filters)):
            module_name = f"vgg_block_{idx+1}"
            self.add_module(
                name=module_name,
                module=VGGBlock(
                    num_convs=nc,
                    in_channels=module_in_channels,
                    out_channels=nf,
                    groups=self.config.groups,
                    **(self.config.block),
                )
            )
            module_in_channels = nf

    def forward(self, input:Tensor) -> Tensor:
        """ finished, checked,

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, n_channels, seq_len)

        Returns
        -------
        output: Tensor,
            of shape (batch_size, n_channels, seq_len)
        """
        output = super().forward(input)
        return output

    def compute_output_shape(self, seq_len:Optional[int]=None, batch_size:Optional[int]=None) -> Sequence[Union[int, None]]:
        """ finished, checked,

        Parameters
        ----------
        seq_len: int,
            length of the 1d sequence
        batch_size: int, optional,
            the batch size, can be None

        Returns
        -------
        output_shape: sequence,
            the output shape of this block, given `seq_len` and `batch_size`
        """
        _seq_len = seq_len
        for module in self:
            output_shape = module.compute_output_shape(_seq_len, batch_size)
            _, _, _seq_len = output_shape
        return output_shape
