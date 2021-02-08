"""
configurations for signal preprocess, feature extraction, training, etc.
along with some constants
"""
import os
from copy import deepcopy

import numpy as np
from easydict import EasyDict as ED

from utils.scoring_aux_data import (
    equiv_class_dict,
    get_class_weight,
)


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ONE_MINUTE_IN_MS = 60 * 1000


# names of the 12 leads
Standard12Leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",]
InferiorLeads = ["II", "III", "aVF",]
LateralLeads = ["I", "aVL",] + [f"V{i}" for i in range(5,7)]
SeptalLeads = ["aVR", "V1",]
AnteriorLeads = [f"V{i}" for i in range(2,5)]
ChestLeads = [f"V{i}" for i in range(1, 7)]
PrecordialLeads = ChestLeads
LimbLeads = ["I", "II", "III", "aVR", "aVL", "aVF",]


# settings from official repo
twelve_leads = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
six_leads = ("I", "II", "III", "aVR", "aVL", "aVF")
three_leads = ("I", "II", "V2")
two_leads = ("II", "V5")



BaseCfg = ED()
BaseCfg.db_dir = "/media/cfs/wenhao71/data/CPSC2021/"
BaseCfg.torch_dtype = "float"  # "double"


# ecg signal preprocessing configurations
PreprocCfg = ED()



SpecialDetectorCfg = ED()



# configurations for visualization
PlotCfg = ED()
# default const for the plot function in dataset.py
# used only when corr. values are absent
# all values are time bias w.r.t. corr. peaks, with units in ms
PlotCfg.p_onset = -40
PlotCfg.p_offset = 40
PlotCfg.q_onset = -20
PlotCfg.s_offset = 40
PlotCfg.qrs_radius = 60
PlotCfg.t_onset = -100
PlotCfg.t_offset = 60



# configurations for building deep learning models
# terminologies of stanford ecg repo. will be adopted
# NOTE: configs of deep learning models have been moved to the folder `model_configs`
ModelCfg = ED()
ModelCfg.fs = 500
ModelCfg.spacing = 1000 / ModelCfg.fs
ModelCfg.bin_pred_thr = 0.5



# training configurations for machine learning and deep learning
TrainCfg = ED()
