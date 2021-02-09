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


__all__ = [
    "BaseCfg",
    "PlotCfg",
    "PreprocCfg",
    "SpecialDetectorCfg",
    "ModelCfg",
    "TrainCfg",
]


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
# PreprocCfg.fs = 500
PreprocCfg.leads_ordering = deepcopy(Standard12Leads)
PreprocCfg.rpeak_mask_radius = 50  # ms
# PreprocCfg.rpeak_num_threshold = 8  # number of leads, used for merging rpeaks detected from 12 leads
PreprocCfg.rpeak_lead_num_thr = 8  # number of leads, used for merging rpeaks detected from 12 leads
PreprocCfg.beat_winL = 250
PreprocCfg.beat_winR = 250



SpecialDetectorCfg = ED()
SpecialDetectorCfg.leads_ordering = deepcopy(PreprocCfg.leads_ordering)
SpecialDetectorCfg.pr_fs_lower_bound = 47  # Hz
SpecialDetectorCfg.pr_spike_mph_ratio = 15  # ratio to the average amplitude of the signal
SpecialDetectorCfg.pr_spike_mpd = 300  # ms
SpecialDetectorCfg.pr_spike_prominence = 0.3
SpecialDetectorCfg.pr_spike_prominence_wlen = 120  # ms
SpecialDetectorCfg.pr_spike_inv_density_threshold = 2500  # inverse density (1/density), one spike per 2000 ms
SpecialDetectorCfg.pr_spike_leads_threshold = 7
SpecialDetectorCfg.axis_qrs_mask_radius = 70  # ms
SpecialDetectorCfg.axis_method = "2-lead"  # can also be "3-lead"
SpecialDetectorCfg.brady_threshold = _ONE_MINUTE_IN_MS / 60  # ms, corr. to 60 bpm
SpecialDetectorCfg.tachy_threshold = _ONE_MINUTE_IN_MS / 100  # ms, corr. to 100 bpm
SpecialDetectorCfg.lqrsv_qrs_mask_radius = 60  # ms
SpecialDetectorCfg.lqrsv_ampl_bias = 0.02  # mV, TODO: should be further determined by resolution, etc.
SpecialDetectorCfg.lqrsv_ratio_threshold = 0.8



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
# `bin_pred_look_again_tol` is used when no prob is greater than `bin_pred_thr`,
# then the prediction would be the one with the highest prob.,
# along with those with prob. no less than the highest prob. minus `bin_pred_look_again_tol`
ModelCfg.bin_pred_look_again_tol = 0.03
ModelCfg.bin_pred_nsr_thr = 0.1
ModelCfg.torch_dtype = BaseCfg.torch_dtype



# training configurations for machine learning and deep learning
TrainCfg = ED()
