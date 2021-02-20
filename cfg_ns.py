"""
configurations for signal preprocess, feature extraction, training, etc.
along with some constants

all classes are treated using the same (deep learning) method uniformly, i.e. no special classes

NEW in CinC2021 compared to CinC2020
"""
import os
from copy import deepcopy

import numpy as np
from easydict import EasyDict as ED

from utils.scoring_aux_data import (
    equiv_class_dict,
    get_class_weight,
)
from torch_ecg.torch_ecg.model_configs import ECG_CRNN_CONFIG


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
BaseCfg.log_dir = os.path.join(_BASE_DIR, "log")
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

# special classes using special detectors
# _SPECIAL_CLASSES = ["Brady", "LAD", "RAD", "PR", "LQRSV"]
_SPECIAL_CLASSES = []



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



# training configurations for machine learning and deep learning
TrainCfg = ED()

# configs of files
TrainCfg.db_dir = BaseCfg.db_dir
TrainCfg.log_dir = BaseCfg.log_dir
TrainCfg.checkpoints = os.path.join(_BASE_DIR, "checkpoints")
TrainCfg.keep_checkpoint_max = 20

TrainCfg.leads = deepcopy(twelve_leads)

# configs of training data
TrainCfg.fs = BaseCfg.fs
TrainCfg.data_format = "channel_first"
TrainCfg.special_classes = deepcopy(_SPECIAL_CLASSES)
TrainCfg.normalize_data = True
TrainCfg.train_ratio = 0.8
TrainCfg.min_class_weight = 0.5
TrainCfg.tranches_for_training = ""  # one of "", "AB", "E", "F"

TrainCfg.tranche_class_weights = ED({
    t: get_class_weight(
        t,
        exclude_classes=TrainCfg.special_classes,
        scored_only=True,
        threshold=20,
        min_weight=TrainCfg.min_class_weight,
    ) for t in ["A", "B", "AB", "E", "F"]
})
TrainCfg.tranche_classes = ED({
    t: sorted(list(t_cw.keys())) \
        for t, t_cw in TrainCfg.tranche_class_weights.items()
})

TrainCfg.class_weights = get_class_weight(
    tranches="ABEF",
    exclude_classes=TrainCfg.special_classes,
    scored_only=True,
    threshold=20,
    min_weight=TrainCfg.min_class_weight,
)
TrainCfg.classes = sorted(list(TrainCfg.class_weights.keys()))

# configs of signal preprocessing
# frequency band of the filter to apply, should be chosen very carefully
# TrainCfg.bandpass = None  # [-np.inf, 45]
# TrainCfg.bandpass = [-np.inf, 45]
TrainCfg.bandpass = [0.5, 60]
TrainCfg.bandpass_order = 5

# configs of data aumentation
TrainCfg.label_smoothing = 0.1
TrainCfg.random_mask = int(TrainCfg.fs * 0.0)  # 1.0s, 0 for no masking
TrainCfg.stretch_compress = 1.0  # stretch or compress in time axis

# configs of training epochs, batch, etc.
TrainCfg.n_epochs = 300
TrainCfg.batch_size = 128
# TrainCfg.max_batches = 500500

# configs of optimizers and lr_schedulers
TrainCfg.train_optimizer = "adam"  # "sgd"

TrainCfg.learning_rate = 0.0001
TrainCfg.lr = TrainCfg.learning_rate
TrainCfg.lr_step_size = 50
TrainCfg.lr_gamma = 0.1

TrainCfg.lr_scheduler = None  # "plateau", "burn_in", "step", None

TrainCfg.burn_in = 400
TrainCfg.steps = [5000, 10000]

TrainCfg.momentum = 0.949
TrainCfg.decay = 0.0005

# configs of loss function
# TrainCfg.loss = "BCEWithLogitsLoss"
TrainCfg.loss = "BCEWithLogitsWithClassWeightLoss"
TrainCfg.log_step = 20
TrainCfg.eval_every = 20

# configs of model selection
TrainCfg.cnn_name = "resnet_leadwise"  # "vgg16", "resnet", "vgg16_leadwise", "cpsc", "cpsc_leadwise"
TrainCfg.rnn_name = "none"  # "none", "lstm", "attention"

# configs of inputs and outputs
# almost all records have duration >= 8s, most have duration >= 10s
# use `utils.utils_signal.ensure_siglen` to ensure signal length
TrainCfg.input_len = int(500 * 10.0)
# tolerance for records with length shorter than `TrainCfg.input_len`
TrainCfg.input_len_tol = int(0.2 * TrainCfg.input_len)
TrainCfg.siglen = TrainCfg.input_len
TrainCfg.bin_pred_thr = ModelCfg.bin_pred_thr
TrainCfg.bin_pred_look_again_tol = ModelCfg.bin_pred_look_again_tol
TrainCfg.bin_pred_nsr_thr = ModelCfg.bin_pred_nsr_thr



# configurations for building deep learning models
# terminologies of stanford ecg repo. will be adopted
ModelCfg = ED()
ModelCfg.torch_dtype = BaseCfg.torch_dtype
ModelCfg.fs = 500
ModelCfg.spacing = 1000 / ModelCfg.fs
ModelCfg.bin_pred_thr = 0.5
# `bin_pred_look_again_tol` is used when no prob is greater than `bin_pred_thr`,
# then the prediction would be the one with the highest prob.,
# along with those with prob. no less than the highest prob. minus `bin_pred_look_again_tol`
ModelCfg.bin_pred_look_again_tol = 0.03
ModelCfg.bin_pred_nsr_thr = 0.1
ModelCfg.special_classes = deepcopy(_SPECIAL_CLASSES)

ModelCfg.dl_classes = deepcopy(TrainCfg.classes)
ModelCfg.dl_siglen = TrainCfg.siglen
ModelCfg.tranche_classes = deepcopy(TrainCfg.tranche_classes)
ModelCfg.full_classes = ModelCfg.dl_classes + ModelCfg.special_classes
ModelCfg.cnn_name = TrainCfg.cnn_name
ModelCfg.rnn_name = TrainCfg.rnn_name


_BASE_MODEL_CONFIG = deepcopy(ECG_CRNN_CONFIG)

# detailed configs for 12-lead, 6-lead, 3-lead, 2-lead models
# mostly follow from torch_ecg.torch_ecg.model_configs.ecg_crnn
ModelCfg.twelve_leads = deepcopy(_BASE_MODEL_CONFIG)
# TODO: add adjustifications for "leadwise" configs for 6,3,2 leads models
ModelCfg.six_leads = ED()
ModelCfg.three_leads = ED()
ModelCfg.two_leads = ED()