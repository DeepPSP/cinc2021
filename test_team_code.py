"""
"""
#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import os, sys
import time
from datetime import datetime
from copy import deepcopy
from logging import Logger
from typing import NoReturn
import traceback
import warnings

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as ED
from scipy.signal import resample, resample_poly

# from torch_ecg.torch_ecg._preprocessors import PreprocManager
from torch_ecg_bak.torch_ecg._preprocessors import PreprocManager

from trainer import CINC2021Trainer
from dataset import CINC2021
# from helper_code import twelve_leads, six_leads, four_leads, three_leads, two_leads, lead_sets
from cfg import (
    TrainCfg, ModelCfg,
    TrainCfg_ns, ModelCfg_ns,
    SpecialDetectorCfg,
)
from model import ECG_CRNN_CINC2021
from utils.special_detectors import special_detectors
from utils.utils_nn import extend_predictions
from utils.misc import rdheader
from utils.utils_signal import ensure_siglen
from utils.scoring_aux_data import abbr_to_snomed_ct_code
from signal_processing.ecg_denoise import remove_spikes_naive

ECG_CRNN_CINC2021.__DEBUG__ = False
CINC2021.__DEBUG__ = False
CINC2021Trainer.__DEBUG__ = False


# Define the Challenge lead sets. These variables are not required. You can change or remove them.
twelve_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6')
six_leads = ('I', 'II', 'III', 'aVR', 'aVL', 'aVF')
four_leads = ('I', 'II', 'III', 'V2')
three_leads = ('I', 'II', 'V2')
two_leads = ('I', 'II')
lead_sets = (twelve_leads, six_leads, four_leads, three_leads, two_leads)


# NOTE: switch between ns and non-ns configs
_TrainCfg = deepcopy(TrainCfg_ns)
_ModelCfg = deepcopy(ModelCfg_ns)
# _TrainCfg = deepcopy(TrainCfg)
# _ModelCfg = deepcopy(ModelCfg)
# _TrainCfg.sig_slice_tol = None
_TrainCfg.bandpass = None


_ModelFilename = {
    n: f"{n}_lead_model.pth.tar" for n in [12, 6, 4, 3, 2,]
}
# twelve_lead_model_filename = "12_lead_model.pth.tar"
# six_lead_model_filename = "6_lead_model.pth.tar"
# three_lead_model_filename = "3_lead_model.pth.tar"
# two_lead_model_filename = "2_lead_model.pth.tar"


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if _ModelCfg.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32

clf_1linear = ED()
clf_1linear.out_channels = [
  # not including the last linear layer, whose out channels equals n_classes
]
clf_1linear.bias = True
clf_1linear.dropouts = 0.0
clf_1linear.activation = "mish"  # for a single layer `SeqLin`, activation is ignored


PPM = PreprocManager.from_config(_TrainCfg)
PPM.rearrange(["bandpass", "normalize"])


################################################################################
#
# Training function
#
################################################################################


# Train your model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    """
    """
    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)
    # os.makedirs(model_directory, exist_ok=True)

    # general configs and logger
    train_config = deepcopy(_TrainCfg)
    train_config.db_dir = data_directory
    train_config.model_dir = model_directory
    train_config.debug = False

    train_config.cnn_name = "resnet_nature_comm_bottle_neck_se"
    train_config.rnn_name = "none"  # "none", "lstm"
    train_config.attn_name = "none"  # "none", "se", "gc", "nl"
    train_config.n_epochs = 2
    train_config.batch_size = 64  # training 12-lead model sometimes requires GPU memory more than 16G (Tesla T4)
    train_config.log_step = 200
    # train_config.max_lr = 1.5e-3
    train_config.early_stopping.patience = 12

    tranches = train_config.tranches_for_training
    if tranches:
        train_classes = train_config.tranche_classes[tranches]
    else:
        train_classes = train_config.classes

    start_time = time.time()

    ds_train_cache = CINC2021(train_config, training=True, lazy=False)
    ds_val_cache = CINC2021(train_config, training=False, lazy=False)

    # Train 12-lead ECG model.
    print("Training 12-lead ECG model...")

    train_config.leads = twelve_leads
    train_config.n_leads = len(train_config.leads)
    train_config.final_model_name = _ModelFilename[12]
    model_config = deepcopy(_ModelCfg.twelve_leads)
    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name
    model_config.clf = clf_1linear

    training_n_leads(train_config, model_config, ds_train_cache, ds_val_cache)


    # Train 6-lead ECG model.
    print("Training 6-lead ECG model...")

    train_config.leads = six_leads
    train_config.n_leads = len(train_config.leads)
    train_config.final_model_name = _ModelFilename[6]
    train_config.batch_size = 64
    model_config = deepcopy(_ModelCfg.six_leads)
    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name
    model_config.clf = clf_1linear

    ds_train = CINC2021.from_extern(ds_train_cache, train_config)
    ds_val = CINC2021.from_extern(ds_val_cache, train_config)

    training_n_leads(train_config, model_config, ds_train, ds_val)

    del ds_train, ds_val


    # Train 4-lead ECG model.
    print("Training 4-lead ECG model...")

    train_config.leads = four_leads
    train_config.n_leads = len(train_config.leads)
    train_config.final_model_name = _ModelFilename[4]
    train_config.batch_size = 64
    model_config = deepcopy(_ModelCfg.four_leads)
    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name
    model_config.clf = clf_1linear

    ds_train = CINC2021.from_extern(ds_train_cache, train_config)
    ds_val = CINC2021.from_extern(ds_val_cache, train_config)

    training_n_leads(train_config, model_config, ds_train, ds_val)

    del ds_train, ds_val
    

    # Train 3-lead ECG model.
    print("Training 3-lead ECG model...")

    train_config.leads = three_leads
    train_config.n_leads = len(train_config.leads)
    train_config.final_model_name = _ModelFilename[3]
    train_config.batch_size = 64
    model_config = deepcopy(_ModelCfg.three_leads)
    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name
    model_config.clf = clf_1linear

    ds_train = CINC2021.from_extern(ds_train_cache, train_config)
    ds_val = CINC2021.from_extern(ds_val_cache, train_config)

    training_n_leads(train_config, model_config, ds_train, ds_val)

    del ds_train, ds_val
    

    # Train 2-lead ECG model.
    print("Training 2-lead ECG model...")

    train_config.leads = two_leads
    train_config.n_leads = len(train_config.leads)
    train_config.final_model_name = _ModelFilename[2]
    train_config.batch_size = 64
    model_config = deepcopy(_ModelCfg.two_leads)
    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name
    model_config.clf = clf_1linear

    ds_train = CINC2021.from_extern(ds_train_cache, train_config)
    ds_val = CINC2021.from_extern(ds_val_cache, train_config)

    training_n_leads(train_config, model_config, ds_train, ds_val)

    del ds_train, ds_val, ds_train_cache, ds_val_cache

    print(f"Training finishes! Total time usage is {((time.time() - start_time) / 3600):.3f} hours.")



def training_n_leads(train_config:ED, model_config:ED, train_dataset:CINC2021, val_dataset:CINC2021) -> NoReturn:
    """
    """
    tranches = train_config.tranches_for_training
    if tranches:
        train_classes = train_config.tranche_classes[tranches]
    else:
        train_classes = train_config.classes
    model = ECG_CRNN_CINC2021(
        classes=train_classes,
        n_leads=train_config.n_leads,
        config=model_config,
    )

    if torch.cuda.device_count() > 1:
        # model = torch.nn.DataParallel(model)
        # test on single GPU, as the official platform only supports single GPU
        pass
    model.to(device=DEVICE)

    trainer = CINC2021Trainer(
        model=model,
        model_config=model_config,
        train_config=train_config,
        device=DEVICE,
        lazy=True,
    )
    train_dataset.to(leads=train_config.leads)
    val_dataset.to(leads=train_config.leads)
    trainer._setup_dataloaders(train_dataset, val_dataset)

    best_state_dict = trainer.train()  # including saving model

    del trainer
    del model
    del best_state_dict

    torch.cuda.empty_cache()



################################################################################
#
# File I/O functions
#
################################################################################

# Save a trained model. This function is not required. You can change or remove it.
def save_model(filename, classes, leads, imputer, classifier):
    # Construct a data structure for the model and save it.
    raise NotImplementedError

# Load a trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def load_model(model_directory, leads):
    n_leads = len(leads)
    model_filename = _ModelFilename[n_leads]
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model, train_cfg = ECG_CRNN_CINC2021.from_checkpoint(
        path=os.path.join(model_directory, model_filename),
    )
    model.eval()
    if len(model.classes) != len(_TrainCfg.classes):
        warnings.warn(f"""checkpoint model has {len(model.classes)} classes, while _TrainCfg has {len(_TrainCfg.classes)}""")
    return model

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def run_model(model, header, recording, verbose=0):
    """ finished, checked,
    """
    raw_data, ann_dict = preprocess_data(header, recording)

    for lead in range(raw_data.shape[0]):
        raw_data[lead, ...] = remove_spikes_naive(raw_data[lead, ...])

    final_scores, final_conclusions = [], []

    if len(_TrainCfg.special_classes) > 0:
        try:
            partial_conclusion = special_detectors(
                raw_data.copy(),
                _TrainCfg.fs,
                sig_fmt="lead_first",
                leads=ann_dict["df_leads"]["lead_name"],
                axis_method="3-lead",
                verbose=verbose
            )
        except Exception as e:
            partial_conclusion = dict(
                is_brady = False,
                is_tachy = False,
                is_LAD = False,
                is_RAD = False,
                is_PR = False,
                is_LQRSV = False,
            )
            print("special_detectors raises errors, as follows")
            traceback.print_exc()

        is_brady = partial_conclusion.is_brady
        is_tachy = partial_conclusion.is_tachy
        is_LAD = partial_conclusion.is_LAD
        is_RAD = partial_conclusion.is_RAD
        is_PR = partial_conclusion.is_PR
        is_LQRSV = partial_conclusion.is_LQRSV

        if verbose >= 1:
            print(f"results from special detectors: {dict_to_str(partial_conclusion)}")

        tmp = np.zeros(shape=(len(_ModelCfg.full_classes,)))
        tmp[_ModelCfg.full_classes.index("Brady")] = int(is_brady)
        tmp[_ModelCfg.full_classes.index("LAD")] = int(is_LAD)
        tmp[_ModelCfg.full_classes.index("RAD")] = int(is_RAD)
        tmp[_ModelCfg.full_classes.index("PR")] = int(is_PR)
        tmp[_ModelCfg.full_classes.index("LQRSV")] = int(is_LQRSV)
        partial_conclusion = tmp

        final_scores.append(partial_conclusion)
        final_conclusions.append(partial_conclusion)
    
    # DL part
    dl_data = raw_data.copy()
    dl_data, _ = PPM(dl_data, fs=ann_dict["fs"])
    # unsqueeze to add a batch dimention
    dl_data = (torch.from_numpy(dl_data)).unsqueeze(0).to(device=DEVICE)

    if "NSR" in _ModelCfg.dl_classes:
        dl_nsr_cid = _ModelCfg.dl_classes.index("NSR")
    elif "426783006" in _ModelCfg.dl_classes:
        dl_nsr_cid = _ModelCfg.dl_classes.index("426783006")
    else:
        dl_nsr_cid = None

    # dl_scores, dl_conclusions each of shape (1,n_classes)
    try:
        dl_scores, dl_conclusions = model.inference(
            dl_data,
            class_names=False,
            bin_pred_thr=0.5
        )
    except Exception as e:
        # in case of errors, set the prediction to "NSR"
        dl_scores = np.zeros(shape=(1, len(_ModelCfg.dl_classes)), dtype=float)
        dl_conclusions = np.zeros(shape=(1, len(_ModelCfg.dl_classes)), dtype=int)
        if dl_nsr_cid is not None:
            ind = dl_nsr_cid
        else:
            ind = np.random.randint(0, len(_ModelCfg.dl_classes))
        dl_scores[0, ind] = 1
        dl_conclusions[0, ind] = 1
    dl_scores = dl_scores[0]
    dl_conclusions = dl_conclusions[0]

    if verbose >= 1:
        print(f"results from dl model:\n{dl_scores}\n{dl_conclusions}")

    if len(_TrainCfg.special_classes) > 0:
        dl_scores = extend_predictions(
            dl_scores,
            _ModelCfg.dl_classes,
            _ModelCfg.full_classes,
        )
        dl_conclusions = extend_predictions(
            dl_conclusions,
            _ModelCfg.dl_classes,
            _ModelCfg.full_classes,
        )

    final_scores.append(dl_scores)
    final_conclusions.append(dl_conclusions)
    final_scores = np.max(final_scores, axis=0)
    final_conclusions = np.max(final_conclusions, axis=0)

    # TODO:
    # filter contradictory conclusions from dl model and from special detector

    classes = _ModelCfg.full_classes
    # class abbr name to snomed ct code
    classes = [abbr_to_snomed_ct_code[c] for c in classes]
    labels = final_conclusions.astype(int).tolist()
    probabilities = final_scores.tolist()

    return classes, labels, probabilities


def preprocess_data(header:str, recording:np.ndarray):
    """
    modified from data_reader.py
    """
    header_data = header.splitlines()
    header_reader = rdheader(header_data)
    ann_dict = {}
    ann_dict["rec_name"], ann_dict["nb_leads"], ann_dict["fs"], ann_dict["nb_samples"], ann_dict["datetime"], daytime = header_data[0].split(" ")

    ann_dict["nb_leads"] = int(ann_dict["nb_leads"])
    ann_dict["fs"] = int(ann_dict["fs"])
    ann_dict["nb_samples"] = int(ann_dict["nb_samples"])
    ann_dict["datetime"] = datetime.strptime(
        " ".join([ann_dict["datetime"], daytime]), "%d-%b-%Y %H:%M:%S"
    )

    df_leads = pd.DataFrame()
    cols = [
        "file_name", "fmt", "byte_offset",
        "adc_gain", "units", "adc_res", "adc_zero",
        "baseline", "init_value", "checksum", "block_size", "sig_name",
    ]
    for k in cols:
        df_leads[k] = header_reader.__dict__[k]
    df_leads = df_leads.rename(columns={"sig_name":"lead_name", "units":"adc_units", "file_name":"filename",})
    df_leads.index = df_leads["lead_name"]
    df_leads.index.name = None
    ann_dict["df_leads"] = df_leads

    header_info = ann_dict["df_leads"]

    data = recording.copy()
    # ensure that data comes in format of "lead_first"
    if data.shape[0] > 12:
        data = data.T
    baselines = header_info["baseline"].values.reshape(data.shape[0], -1)
    adc_gain = header_info["adc_gain"].values.reshape(data.shape[0], -1)
    data = np.asarray(data-baselines) / adc_gain

    if ann_dict["fs"] != _TrainCfg.fs:
        data = resample_poly(data, _TrainCfg.fs, ann_dict["fs"], axis=1)

    return data, ann_dict




def test_model(model_directory, data_directory, output_directory):
    """
    the function from test_model.py
    """
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for the outputs if it does not already exist.
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)

    # Identify the required lead sets.
    required_lead_sets = set()
    for i in range(num_recordings):
        header = load_header(header_files[i])
        leads = get_leads(header)
        sorted_leads = sort_leads(leads)
        required_lead_sets.add(sorted_leads)

    # Load models.
    leads_to_model = dict()
    print('Loading models...')
    for leads in required_lead_sets:
        model = load_model(model_directory, leads) ### Implement this function!
        leads_to_model[leads] = model

    # Run model for each recording.
    print('Running model...')

    for i in range(num_recordings):
        print('    {}/{}...'.format(i+1, num_recordings))

        # Load header and recording.
        header = load_header(header_files[i])
        recording = load_recording(recording_files[i])
        leads = get_leads(header)
        sorted_leads = sort_leads(leads)

        # Apply model to recording.
        model = leads_to_model[sorted_leads]
        try:
            classes, labels, probabilities = run_model(model, header, recording) ### Implement this function!
        except Exception as e:
            print(e)
            print('... failed.')
            classes, labels, probabilities = list(), list(), list()

        # Save model outputs.
        recording_id = get_recording_id(header)
        head, tail = os.path.split(header_files[i])
        root, extension = os.path.splitext(tail)
        output_file = os.path.join(output_directory, root + '.csv')
        save_outputs(output_file, recording_id, classes, labels, probabilities)

    print('Done.')




BASE_DIR = os.path.dirname(TrainCfg_ns.model_dir)
TRAINING_DATA_DIR = os.path.join(BASE_DIR, "docker_test_dir", "training_data")
MODEL_DIR = os.path.join(BASE_DIR, "docker_test_dir", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
TEST_DATA_DIR = os.path.join(BASE_DIR, "docker_test_dir", "test_data")
TEST_OUTPUT_DIR = os.path.join(BASE_DIR, "docker_test_dir", "test_output")
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)


def test_load_model() -> NoReturn:
    """
    """
    for l in lead_sets:
        try:
            model = load_model(MODEL_DIR, l)
        except:
            model = load_model(TrainCfg_ns.model_dir, l)
        print(f"n_leads = {len(l)}")
        print(model)
        print("#"*80 + "\n\n")


def test_run_model() -> NoReturn:
    """
    """
    for l in ["twelve", "six", "four", "three", "two",]:
        try:
            test_model(
                MODEL_DIR,
                os.path.join(TEST_DATA_DIR, f"{l}_leads"),
                os.path.join(TEST_OUTPUT_DIR, f"test_results_{l}_leads"),
            )
        except:
            test_model(
                TrainCfg_ns.model_dir,
                os.path.join(TEST_DATA_DIR, f"{l}_leads"),
                os.path.join(TEST_OUTPUT_DIR, f"test_results_{l}_leads"),
            )


if __name__ == "__main__":
    print("#"*80)
    print("train models...")
    training_code(TRAINING_DATA_DIR, MODEL_DIR)
    print("#"*80)
    print("test_load_model...")
    test_load_model()
    print("\n"*3)
    print("#"*80)
    print("test_run_model...")
    test_run_model()
