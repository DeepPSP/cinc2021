"""
validated C(R)NN structure models,
for classifying ECG arrhythmias
"""
from copy import deepcopy
from itertools import repeat
from collections import OrderedDict
from typing import Union, Optional, Tuple, Sequence, NoReturn, Any
from numbers import Real, Number

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import pandas as pd
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from easydict import EasyDict as ED

from ..cfg import DEFAULTS
from ..model_configs.ecg_crnn import ECG_CRNN_CONFIG
from ..utils.utils_nn import (
    compute_conv_output_shape, compute_module_size,
    SizeMixin,
)
from ..utils.misc import dict_to_str
from ._nets import (
    Mish, Swish, Activations,
    Bn_Activation, Conv_Bn_Activation,
    DownSample,
    ZeroPadding,
    StackedLSTM,
    AttentionWithContext,
    SelfAttention, MultiHeadAttention,
    AttentivePooling,
    NonLocalBlock, SEBlock, GlobalContextBlock,
    SeqLin,
)
from .cnn.vgg import VGG16
from .cnn.resnet import ResNet
from .cnn.multi_scopic import MultiScopicCNN
from .cnn.densenet import DenseNet
from .cnn.xception import Xception
# from .cnn import (
#     VGG16, ResNet, MultiScopicCNN, DenseNet, Xception,
# )
from .transformers import Transformer


if DEFAULTS.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "ECG_CRNN",
]


class ECG_CRNN(SizeMixin, nn.Module):
    """ finished, continuously improving,

    C(R)NN models modified from the following refs.

    References
    ----------
    [1] Yao, Qihang, et al. "Time-Incremental Convolutional Neural Network for Arrhythmia Detection in Varied-Length Electrocardiogram." 2018 IEEE 16th Intl Conf on Dependable, Autonomic and Secure Computing, 16th Intl Conf on Pervasive Intelligence and Computing, 4th Intl Conf on Big Data Intelligence and Computing and Cyber Science and Technology Congress (DASC/PiCom/DataCom/CyberSciTech). IEEE, 2018.
    [2] Yao, Qihang, et al. "Multi-class Arrhythmia detection from 12-lead varied-length ECG using Attention-based Time-Incremental Convolutional Neural Network." Information Fusion 53 (2020): 174-182.
    [3] Hannun, Awni Y., et al. "Cardiologist-level arrhythmia detection and classification in ambulatory electrocardiograms using a deep neural network." Nature medicine 25.1 (2019): 65.
    [4] https://stanfordmlgroup.github.io/projects/ecg2/
    [5] https://github.com/awni/ecg
    [6] CPSC2018 entry 0236
    [7] CPSC2019 entry 0416
    """
    __DEBUG__ = False
    __name__ = "ECG_CRNN"

    def __init__(self, classes:Sequence[str], n_leads:int, config:Optional[ED]=None, **kwargs:Any) -> NoReturn:
        """ finished, checked,

        Parameters
        ----------
        classes: list,
            list of the classes for classification
        n_leads: int,
            number of leads (number of input channels)
        config: dict, optional,
            other hyper-parameters, including kernel sizes, etc.
            ref. the corresponding config file
        """
        super().__init__()
        self.classes = list(classes)
        self.n_classes = len(classes)
        self.n_leads = n_leads
        self.config = deepcopy(ECG_CRNN_CONFIG)
        self.config.update(deepcopy(config) or {})
        if self.__DEBUG__:
            print(f"classes (totally {self.n_classes}) for prediction:{self.classes}")
            print(f"configuration of {self.__name__} is as follows\n{dict_to_str(self.config)}")
            debug_input_len = 4000
        
        cnn_choice = self.config.cnn.name.lower()
        if "vgg16" in cnn_choice:
            self.cnn = VGG16(self.n_leads, **(self.config.cnn[self.config.cnn.name]))
            # rnn_input_size = self.config.cnn.vgg16.num_filters[-1]
        elif "resnet" in cnn_choice:
            self.cnn = ResNet(self.n_leads, **(self.config.cnn[self.config.cnn.name]))
            # rnn_input_size = \
            #     2**len(self.config.cnn[cnn_choice].num_blocks) * self.config.cnn[cnn_choice].init_num_filters
        elif "multi_scopic" in cnn_choice:
            self.cnn = MultiScopicCNN(self.n_leads, **(self.config.cnn[self.config.cnn.name]))
            # rnn_input_size = self.cnn.compute_output_shape(None, None)[1]
        elif "densenet" in cnn_choice or "dense_net" in cnn_choice:
            self.cnn = DenseNet(self.n_leads, **(self.config.cnn[self.config.cnn.name]))
        else:
            raise NotImplementedError(f"the CNN \042{cnn_choice}\042 not implemented yet")
        rnn_input_size = self.cnn.compute_output_shape(None, None)[1]

        if self.__DEBUG__:
            cnn_output_shape = self.cnn.compute_output_shape(debug_input_len, None)
            print(f"cnn output shape (batch_size, features, seq_len) = {cnn_output_shape}, given input_len = {debug_input_len}")

        if self.config.rnn.name.lower() == "none":
            self.rnn = None
            attn_input_size = rnn_input_size
        elif self.config.rnn.name.lower() == "lstm":
            # hidden_sizes = self.config.rnn.lstm.hidden_sizes + [self.n_classes]
            # if self.__DEBUG__:
            #     print(f"lstm hidden sizes {self.config.rnn.lstm.hidden_sizes} ---> {hidden_sizes}")
            self.rnn = StackedLSTM(
                input_size=rnn_input_size,
                hidden_sizes=self.config.rnn.lstm.hidden_sizes,
                bias=self.config.rnn.lstm.bias,
                dropouts=self.config.rnn.lstm.dropouts,
                bidirectional=self.config.rnn.lstm.bidirectional,
                return_sequences=self.config.rnn.lstm.retseq,
            )
            attn_input_size = self.rnn.compute_output_shape(None, None)[-1]
        elif self.config.rnn.name.lower() == "linear":
            self.rnn = SeqLin(
                in_channels=rnn_input_size,
                out_channels=self.config.rnn.linear.out_channels,
                activation=self.config.rnn.linear.activation,
                bias=self.config.rnn.linear.bias,
                dropouts=self.config.rnn.linear.dropouts,
            )
            attn_input_size = self.rnn.compute_output_shape(None, None)[-1]
        else:
            raise NotImplementedError

        # attention
        if self.config.rnn.name.lower() == "lstm" and not self.config.rnn.lstm.retseq:
            self.attn = None
            clf_input_size = attn_input_size
            if self.config.attn.name.lower() != "none":
                print(f"since `retseq` of rnn is False, hence attention `{self.config.attn.name}` is ignored")
        elif self.config.attn.name.lower() == "none":
            self.attn = None
            clf_input_size = attn_input_size
        elif self.config.attn.name.lower() == "nl":  # non_local
            self.attn = NonLocalBlock(
                in_channels=attn_input_size,
                filter_lengths=self.config.attn.nl.filter_lengths,
                subsample_length=self.config.attn.nl.subsample_length,
                batch_norm=self.config.attn.nl.batch_norm,
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[1]
        elif self.config.attn.name.lower() == "se":  # squeeze_exitation
            self.attn = SEBlock(
                in_channels=attn_input_size,
                reduction=self.config.attn.se.reduction,
                activation=self.config.attn.se.activation,
                kw_activation=self.config.attn.se.kw_activation,
                bias=self.config.attn.se.bias,
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[1]
        elif self.config.attn.name.lower() == "gc":  # global_context
            self.attn = GlobalContextBlock(
                in_channels=attn_input_size,
                ratio=self.config.attn.gc.ratio,
                reduction=self.config.attn.gc.reduction,
                pooling_type=self.config.attn.gc.pooling_type,
                fusion_types=self.config.attn.gc.fusion_types,
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[1]
        elif self.config.attn.name.lower() == "sa":  # self_attention
            # NOTE: this branch NOT tested
            self.attn = SelfAttention(
                in_features=attn_input_size,
                head_num=self.config.attn.sa.head_num,
                dropout=self.config.attn.sa.dropout,
                bias=self.config.attn.sa.bias,
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[-1]
        elif self.config.attn.name.lower() == "transformer":
            self.attn = Transformer(
                input_size=attn_input_size,
                hidden_size=self.config.attn.transformer.hidden_size,
                num_layers=self.config.attn.transformer.num_layers,
                num_heads=self.config.attn.transformer.num_heads,
                dropout=self.config.attn.transformer.dropout,
                activation=self.config.attn.transformer.activation,
            )
            clf_input_size = self.attn.compute_output_shape(None, None)[-1]
        else:
            raise NotImplementedError

        if self.__DEBUG__:
            print(f"clf_input_size = {clf_input_size}")

        if self.config.rnn.name.lower() == "lstm" and not self.config.rnn.lstm.retseq:
            self.pool = None
            if self.config.global_pool.lower() != "none":
                print(f"since `retseq` of rnn is False, hence global pooling `{self.config.global_pool}` is ignored")
        elif self.config.global_pool.lower() == "max":
            self.pool = nn.AdaptiveMaxPool1d((1,), return_indices=False)
        elif self.config.global_pool.lower() == "avg":
            self.pool = nn.AdaptiveAvgPool1d((1,))
        elif self.config.global_pool.lower() == "attn":
            raise NotImplementedError("Attentive pooling not implemented yet!")
        else:
            raise NotImplementedError(f"pooling method {self.config.global_pool} not implemented yet!")

        # input of `self.clf` has shape: batch_size, channels
        self.clf = SeqLin(
            in_channels=clf_input_size,
            out_channels=self.config.clf.out_channels + [self.n_classes],
            activation=self.config.clf.activation,
            bias=self.config.clf.bias,
            dropouts=self.config.clf.dropouts,
            skip_last_activation=True,
        )

        # sigmoid for inference
        self.sigmoid = nn.Sigmoid()  # for making inference

    def extract_features(self, input:Tensor) -> Tensor:
        """ finished, checked,

        extract feature map before the dense (linear) classifying layer(s)

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, channels, seq_len)
        
        Returns
        -------
        features: Tensor,
            of shape (batch_size, channels, seq_len) or (batch_size, channels)
        """
        # CNN
        features = self.cnn(input)  # batch_size, channels, seq_len
        # print(f"cnn out shape = {features.shape}")

        # RNN (optional)
        if self.config.rnn.name.lower() in ["lstm"]:
            # (batch_size, channels, seq_len) --> (seq_len, batch_size, channels)
            features = features.permute(2,0,1)
            features = self.rnn(features)  # (seq_len, batch_size, channels) or (batch_size, channels)
        elif self.config.rnn.name.lower() in ["linear"]:
            # (batch_size, channels, seq_len) --> (batch_size, seq_len, channels)
            features = features.permute(0,2,1)
            features = self.rnn(features)  # (batch_size, seq_len, channels)
            # (batch_size, seq_len, channels) --> (seq_len, batch_size, channels)
            features = features.permute(1,0,2)
        else:
            # (batch_size, channels, seq_len) --> (seq_len, batch_size, channels)
            features = features.permute(2,0,1)

        # Attention (optional)
        if self.attn is None and features.ndim == 3:
            # (seq_len, batch_size, channels) --> (batch_size, channels, seq_len)
            features = features.permute(1,2,0)
        elif self.config.attn.name.lower() in ["nl", "se", "gc"]:
            # (seq_len, batch_size, channels) --> (batch_size, channels, seq_len)
            features = features.permute(1,2,0)
            features = self.attn(features)  # (batch_size, channels, seq_len)
        elif self.config.attn.name.lower() in ["sa"]:
            features = self.attn(features)  # (seq_len, batch_size, channels)
            # (seq_len, batch_size, channels) -> (batch_size, channels, seq_len)
            features = features.permute(1,2,0)
        elif self.config.attn.name.lower() in ["transformer"]:
            features = self.attn(features)
            # (seq_len, batch_size, channels) -> (batch_size, channels, seq_len)
            features = features.permute(1,2,0)
        return features

    def forward(self, input:Tensor) -> Tensor:
        """ finished, partly checked (rnn part might have bugs),

        Parameters
        ----------
        input: Tensor,
            of shape (batch_size, channels, seq_len)
        
        Returns
        -------
        pred: Tensor,
            of shape (batch_size, n_classes)
        """
        features = self.extract_features(input)

        if self.pool:
            features = self.pool(features)  # (batch_size, channels, 1)
            features = features.squeeze(dim=-1)
        else:
            # features of shape (batch_size, channels)
            pass

        # print(f"clf in shape = {x.shape}")
        pred = self.clf(features)  # batch_size, n_classes

        return pred

    @torch.no_grad()
    def inference(self, input:Union[np.ndarray,Tensor], class_names:bool=False, bin_pred_thr:float=0.5) -> Tuple[Union[np.ndarray, pd.DataFrame], np.ndarray]:
        """ finished, checked,

        Parameters
        ----------
        input: ndarray or Tensor,
            input tensor, of shape (batch_size, channels, seq_len)
        class_names: bool, default False,
            if True, the returned scalar predictions will be a `DataFrame`,
            with class names for each scalar prediction
        bin_pred_thr: float, default 0.5,
            the threshold for making binary predictions from scalar predictions

        Returns
        -------
        pred: ndarray or DataFrame,
            scalar predictions, (and binary predictions if `class_names` is True)
        bin_pred: ndarray,
            the array (with values 0, 1 for each class) of binary prediction
        """
        raise NotImplementedError(f"implement a task specific inference method")

    @staticmethod
    def from_checkpoint(path:str, device:Optional[torch.device]=None) -> Tuple[nn.Module, dict]:
        """

        Parameters
        ----------
        path: str,
            path of the checkpoint
        device: torch.device, optional,
            map location of the model parameters,
            defaults "cuda" if available, otherwise "cpu"

        Returns
        -------
        model: Module,
            the model loaded from a checkpoint
        aux_config: dict,
            auxiliary configs that are needed for data preprocessing, etc.
        """
        _device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        ckpt = torch.load(path, map_location=_device)
        aux_config = ckpt.get("train_config", None) or ckpt.get("config", None)
        assert aux_config is not None, "input checkpoint has no sufficient data to recover a model"
        model = ECG_CRNN(
            classes=aux_config["classes"],
            n_leads=aux_config["n_leads"],
            config=ckpt["model_config"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        return model, aux_config
