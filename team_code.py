#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of required functions, remove non-required functions, and add your own function.

from helper_code import *
import numpy as np, os, sys, joblib
import time
from datetime import datetime
from copy import deepcopy
from logging import Logger
from typing import NoReturn
import traceback

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as ED
from scipy.signal import resample, resample_poly

from train import train
from cfg import TrainCfg, ModelCfg, SpecialDetectorCfg
from cfg import TrainCfg as TrainCfg_ns, ModelCfg as ModelCfg_ns
from model import ECG_CRNN_CINC2021
from utils.special_detectors import special_detectors
from utils.utils_nn import extend_predictions
from utils.misc import get_date_str, dict_to_str, init_logger, rdheader
from utils.utils_signal import ensure_siglen, butter_bandpass_filter
from utils.scoring_aux_data import abbr_to_snomed_ct_code


twelve_lead_model_filename = "12_lead_model.pth.tar"
six_lead_model_filename = "6_lead_model.pth.tar"
three_lead_model_filename = "3_lead_model.pth.tar"
two_lead_model_filename = "2_lead_model.pth.tar"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if ModelCfg.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)
    DTYPE = np.float64
else:
    DTYPE = np.float32


################################################################################
#
# Training function
#
################################################################################


# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    print("Finding header and recording files...")

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception("No data was provided.")

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)
    # os.makedirs(model_directory, exist_ok=True)

    # Extract classes from dataset.
    print("Extracting classes...")

    classes = set()
    for header_file in header_files:
        header = load_header(header_file)
        classes |= set(get_labels(header))
    if all(is_integer(x) for x in classes):
        classes = sorted(classes, key=lambda x: int(x)) # Sort classes numerically if numbers.
    else:
        classes = sorted(classes) # Sort classes alphanumerically otherwise.
    num_classes = len(classes)


    # configs and logger
    train_config = deepcopy(TrainCfg)
    train_config.db_dir = data_directory
    train_config.model_dir = model_directory
    train_config.debug = False
    train_config.n_epochs = 30

    tranches = train_config.tranches_for_training
    if tranches:
        train_classes = train_config.tranche_classes[tranches]
    else:
        train_classes = train_config.classes

    logger = init_logger(log_dir=train_config.log_dir, verbose=2)
    logger.info(f"\n{'*'*20}   Start Training   {'*'*20}\n")
    logger.info(f"Using device {DEVICE}")
    logger.info(f"Using torch of version {torch.__version__}")
    # logger.info(f"with configuration\n{dict_to_str(train_config)}")

    start_time = time.time()


    # Train 12-lead ECG model.
    print("Training 12-lead ECG model...")

    train_config.leads = twelve_leads
    train_config.n_leads = len(train_config.leads)
    train_config.final_model_name = twelve_lead_model_filename
    model_config = deepcopy(ModelCfg.twelve_leads)
    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name

    training_12_leads(train_config, model_config, logger)


    # Train 6-lead ECG model.
    print("Training 6-lead ECG model...")

    train_config.leads = six_leads
    train_config.n_leads = len(train_config.leads)
    train_config.final_model_name = six_lead_model_filename
    model_config = deepcopy(ModelCfg.six_leads)
    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name

    training_6_leads(train_config, model_config, logger)
    

    # Train 3-lead ECG model.
    print("Training 3-lead ECG model...")

    train_config.leads = three_leads
    train_config.n_leads = len(train_config.leads)
    train_config.final_model_name = three_lead_model_filename
    model_config = deepcopy(ModelCfg.three_leads)
    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name

    training_3_leads(train_config, model_config, logger)
    

    # Train 2-lead ECG model.
    print("Training 2-lead ECG model...")

    train_config.leads = two_leads
    train_config.n_leads = len(train_config.leads)
    train_config.final_model_name = two_lead_model_filename
    model_config = deepcopy(ModelCfg.two_leads)
    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name

    training_2_leads(train_config, model_config, logger)

    print(f"Training finishes! Total time usage is {((time.time() - start_time) / 3600):.3f} hours.")



def training_12_leads(train_config:ED, model_config:ED, logger:Logger) -> NoReturn:
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
        # input_len=config.input_len,
        config=model_config,
    )

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.to(device=DEVICE)
    model.__DEBUG__ = False

    train(
        model=model,
        model_config=model_config,
        config=train_config,
        device=DEVICE,
        logger=logger,
        debug=train_config.debug,
    )


def training_6_leads(train_config:ED, model_config:ED, logger:Logger) -> NoReturn:
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
        model = torch.nn.DataParallel(model)
    model.to(device=DEVICE)
    model.__DEBUG__ = False

    train(
        model=model,
        model_config=model_config,
        config=train_config,
        device=DEVICE,
        logger=logger,
        debug=train_config.debug,
    )


def training_3_leads(train_config:ED, model_config:ED, logger:Logger) -> NoReturn:
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
        model = torch.nn.DataParallel(model)
    model.to(device=DEVICE)
    model.__DEBUG__ = False

    train(
        model=model,
        model_config=model_config,
        config=train_config,
        device=DEVICE,
        logger=logger,
        debug=train_config.debug,
    )


def training_2_leads(train_config:ED, model_config:ED, logger:Logger) -> NoReturn:
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
        model = torch.nn.DataParallel(model)
    model.to(device=DEVICE)
    model.__DEBUG__ = False

    train(
        model=model,
        model_config=model_config,
        config=train_config,
        device=DEVICE,
        logger=logger,
        debug=train_config.debug,
    )



################################################################################
#
# File I/O functions
#
################################################################################

# Save your trained models.
def save_model(filename, classes, leads, imputer, classifier):
    # Construct a data structure for the model and save it.
    raise NotImplementedError

# Load your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_twelve_lead_model(model_directory):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    ckpt = torch.load(os.path.join(model_directory, twelve_lead_model_filename), map_location=device)
    model = ECG_CRNN_CINC2021(
        classes=ckpt["train_config"].classes,
        n_leads=12,  # ckpt["train_config"].n_leads
        config=ckpt["model_config"],
    )
    model.eval()
    model.load_state_dict(ckpt["model_state_dict"])
    return model

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    ckpt = torch.load(os.path.join(model_directory, six_lead_model_filename), map_location=device)
    model = ECG_CRNN_CINC2021(
        classes=ckpt["train_config"].classes,
        n_leads=6,  # ckpt["train_config"].n_leads
        config=ckpt["model_config"],
    )
    model.eval()
    model.load_state_dict(ckpt["model_state_dict"])
    return model

# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    ckpt = torch.load(os.path.join(model_directory, three_lead_model_filename), map_location=device)
    model = ECG_CRNN_CINC2021(
        classes=ckpt["train_config"].classes,
        n_leads=3,  # ckpt["train_config"].n_leads
        config=ckpt["model_config"],
    )
    model.eval()
    model.load_state_dict(ckpt["model_state_dict"])
    return model

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    ckpt = torch.load(os.path.join(model_directory, two_lead_model_filename), map_location=device)
    model = ECG_CRNN_CINC2021(
        classes=ckpt["train_config"].classes,
        n_leads=2,  # ckpt["train_config"].n_leads
        config=ckpt["model_config"],
    )
    model.eval()
    model.load_state_dict(ckpt["model_state_dict"])
    return model

# Generic function for loading a model.
def load_model(filename):
    raise NotImplementedError

################################################################################
#
# Running trained model functions
#
################################################################################

# Run your trained 12-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_twelve_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_six_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_three_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Run your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def run_two_lead_model(model, header, recording):
    return run_model(model, header, recording)

# Generic function for running a trained model.
def run_model(model, header, recording, verbose=0):
    raw_data, ann_dict = preprocess_data(header, recording)

    final_scores, final_conclusions = [], []

    if len(TrainCfg.special_classes) > 0:
        try:
            partial_conclusion = special_detectors(
                raw_data.copy(),
                TrainCfg.fs,
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

        tmp = np.zeros(shape=(len(ModelCfg.full_classes,)))
        tmp[ModelCfg.full_classes.index("Brady")] = int(is_brady)
        tmp[ModelCfg.full_classes.index("LAD")] = int(is_LAD)
        tmp[ModelCfg.full_classes.index("RAD")] = int(is_RAD)
        tmp[ModelCfg.full_classes.index("PR")] = int(is_PR)
        tmp[ModelCfg.full_classes.index("LQRSV")] = int(is_LQRSV)
        partial_conclusion = tmp

        final_scores.append(partial_conclusion)
        final_conclusions.append(partial_conclusion)
    
    # DL part
    dl_data = raw_data.copy()
    if TrainCfg.bandpass is not None:
        # bandpass
        dl_data = butter_bandpass_filter(
            dl_data,
            lowcut=TrainCfg.bandpass[0],
            highcut=TrainCfg.bandpass[1],
            order=TrainCfg.bandpass_order,
            fs=TrainCfg.fs,
        )
    # if dl_data.shape[1] >= ModelCfg.dl_siglen:
    #     dl_data = ensure_siglen(dl_data, siglen=ModelCfg.dl_siglen, fmt="lead_first")
    #     if TrainCfg.normalize_data:
    #         # normalize
    #         dl_data = ((dl_data - np.mean(dl_data)) / np.std(dl_data)).astype(DTYPE)
    # else:
    #     if TrainCfg.normalize_data:
    #         # normalize
    #         dl_data = ((dl_data - np.mean(dl_data)) / np.std(dl_data)).astype(DTYPE)
    #     dl_data = ensure_siglen(dl_data, siglen=ModelCfg.dl_siglen, fmt="lead_first")
    if TrainCfg.normalize_data:
        # normalize
        dl_data = ((dl_data - np.mean(dl_data)) / np.std(dl_data)).astype(DTYPE)
    # unsqueeze to add a batch dimention
    dl_data = (torch.from_numpy(dl_data)).unsqueeze(0).to(device=DEVICE)

    if "NSR" in ModelCfg.dl_classes:
        dl_nsr_cid = ModelCfg.dl_classes.index("NSR")
    elif "426783006" in ModelCfg.dl_classes:
        dl_nsr_cid = ModelCfg.dl_classes.index("426783006")
    else:
        dl_nsr_cid = None

    # dl_scores, dl_conclusions each of shape (1,n_classes)
    dl_scores, dl_conclusions = model.inference(
        dl_data,
        class_names=False,
        bin_pred_thr=0.5
    )
    dl_scores = dl_scores[0]
    dl_conclusions = dl_conclusions[0]

    if verbose >= 1:
        print(f"results from dl model:\n{dl_scores}\n{dl_conclusions}")

    dl_scores = extend_predictions(
        dl_scores,
        ModelCfg.dl_classes,
        ModelCfg.full_classes,
    )
    dl_conclusions = extend_predictions(
        dl_conclusions,
        ModelCfg.dl_classes,
        ModelCfg.full_classes,
    )

    final_scores.append(dl_scores)
    final_conclusions.append(dl_conclusions)
    final_scores = np.max(final_scores, axis=0)
    final_conclusions = np.max(final_conclusions, axis=0)

    # TODO:
    # filter contradictory conclusions from dl model and from special detector


    classes = ModelCfg.full_classes
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
    # try:
    #     ann_dict["age"] = \
    #         int([l for l in header_reader.comments if "Age" in l][0].split(": ")[-1])
    # except:
    #     ann_dict["age"] = np.nan
    # try:
    #     ann_dict["sex"] = \
    #         [l for l in header_reader.comments if "Sex" in l][0].split(": ")[-1]
    # except:
    #     ann_dict["sex"] = "Unknown"
    # try:
    #     ann_dict["medical_prescription"] = \
    #         [l for l in header_reader.comments if "Rx" in l][0].split(": ")[-1]
    # except:
    #     ann_dict["medical_prescription"] = "Unknown"
    # try:
    #     ann_dict["history"] = \
    #         [l for l in header_reader.comments if "Hx" in l][0].split(": ")[-1]
    # except:
    #     ann_dict["history"] = "Unknown"
    # try:
    #     ann_dict["symptom_or_surgery"] = \
    #         [l for l in header_reader.comments if "Sx" in l][0].split(": ")[-1]
    # except:
    #     ann_dict["symptom_or_surgery"] = "Unknown"

    # l_Dx = [l for l in header_reader.comments if "Dx" in l][0].split(": ")[-1].split(",")
    # ann_dict["diagnosis"], ann_dict["diagnosis_scored"] = self._parse_diagnosis(l_Dx)

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

    if ann_dict["fs"] != TrainCfg.fs:
        data = resample_poly(data, TrainCfg.fs, ann_dict["fs"], axis=1)

    return data, ann_dict
