"""
"""

from typing import NoReturn, Optional, Tuple, Sequence

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from easydict import EasyDict as ED

# from torch_ecg.torch_ecg.utils.nas import NAS
from torch_ecg_bak.torch_ecg.utils.nas import NAS

from trainer import CINC2021Trainer
from dataset import CINC2021
from model import ECG_CRNN_CINC2021


__all__ = ["CINC2021NAS"]


class CINC2021NAS(NAS):
    """
    """
    __name__ = "CINC2021NAS"

    def __init__(self,
                 train_config:dict,
                 model_configs:Sequence[dict],
                 lazy:bool=False,) -> NoReturn:
        """ finished, NOT checked,

        Parameters
        ----------
        train_config: dict,
            train configurations
        model_configs: sequence of dict,
            model configurations, each with a different network architecture
        """
        super().__init__(
            CINC2021Trainer, ECG_CRNN_CINC2021, CINC2021,
            train_config, model_configs, lazy,
        )
