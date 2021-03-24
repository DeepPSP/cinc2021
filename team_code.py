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

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as ED
from scipy.signal import resample, resample_poly

from train import train
from cfg import TrainCfg, ModelCfg, SpecialDetectorCfg
from model import ECG_CRNN_CINC2021
from utils.special_detectors import special_detectors
from utils.utils_nn import extend_predictions
from utils.misc import get_date_str, dict_to_str, init_logger, rdheader
from utils.utils_signal import ensure_siglen, butter_bandpass_filter


twelve_lead_model_filename = '12_lead_model.pth'
six_lead_model_filename = '6_lead_model.pth'
three_lead_model_filename = '3_lead_model.pth'
two_lead_model_filename = '2_lead_model.pth'


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


################################################################################
#
# Training function
#
################################################################################


# Train your model. This function is *required*. Do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    # Find header and recording files.
    print('Finding header and recording files...')

    header_files, recording_files = find_challenge_files(data_directory)
    num_recordings = len(recording_files)

    if not num_recordings:
        raise Exception('No data was provided.')

    # Create a folder for the model if it does not already exist.
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    # Extract classes from dataset.
    print('Extracting classes...')

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
    train_config.debug = True

    tranches = train_config.tranches_for_training
    if tranches:
        train_classes = train_config.tranche_classes[tranches]
    else:
        train_classes = train_config.classes

    logger = init_logger(log_dir=train_config.log_dir, verbose=2)
    logger.info(f"\n{'*'*20}   Start Training   {'*'*20}\n")
    logger.info(f"Using device {DEVICE}")
    logger.info(f"Using torch of version {torch.__version__}")
    logger.info(f"with configuration\n{dict_to_str(train_config)}")

    start_time = time.time()


    # Train 12-lead ECG model.
    print('Training 12-lead ECG model...')

    train_config.leads = twelve_leads
    train_config.n_leads = len(train_config.leads)
    # filename = os.path.join(model_directory, twelve_lead_model_filename)
    train_config.final_model_name = twelve_lead_model_filename
    model_config = deepcopy(ModelCfg.twelve_leads)
    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name

    training_12_leads(train_config, model_config, logger)


    # Train 6-lead ECG model.
    print('Training 6-lead ECG model...')

    train_config.leads = six_leads
    train_config.n_leads = len(train_config.leads)
    train_config.final_model_name = six_lead_model_filename
    model_config = deepcopy(ModelCfg.six_leads)
    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name

    training_6_leads(train_config, model_config, logger)
    

    # Train 3-lead ECG model.
    print('Training 3-lead ECG model...')

    train_config.leads = three_leads
    train_config.n_leads = len(train_config.leads)
    train_config.final_model_name = three_lead_model_filename
    model_config = deepcopy(ModelCfg.three_leads)
    model_config.cnn.name = train_config.cnn_name
    model_config.rnn.name = train_config.rnn_name
    model_config.attn.name = train_config.attn_name

    training_3_leads(train_config, model_config, logger)
    

    # Train 2-lead ECG model.
    print('Training 2-lead ECG model...')

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
    model = ECG_CRNN_CINC2021(
        classes=ModelCfg.dl_classes,
        n_leads=12,
        config=ModelCfg.twelve_leads,
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.eval()
    model.load_state_dict(torch.load(os.path.join(TrainCfg.model_dir, twelve_lead_model_filename), map_location=device))
    return model

# Load your trained 6-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_six_lead_model(model_directory):
    model = ECG_CRNN_CINC2021(
        classes=ModelCfg.dl_classes,
        n_leads=6,
        config=ModelCfg.six_leads,
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.eval()
    model.load_state_dict(torch.load(os.path.join(TrainCfg.model_dir, six_lead_model_filename), map_location=device))
    return model

# Load your trained 3-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_three_lead_model(model_directory):
    model = ECG_CRNN_CINC2021(
        classes=ModelCfg.dl_classes,
        n_leads=3,
        config=ModelCfg.three_leads,
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.eval()
    model.load_state_dict(torch.load(os.path.join(TrainCfg.model_dir, three_lead_model_filename), map_location=device))
    return model

# Load your trained 2-lead ECG model. This function is *required*. Do *not* change the arguments of this function.
def load_two_lead_model(model_directory):
    model = ECG_CRNN_CINC2021(
        classes=ModelCfg.dl_classes,
        n_leads=2,
        config=ModelCfg.two_leads,
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model.eval()
    model.load_state_dict(torch.load(os.path.join(TrainCfg.model_dir, two_lead_model_filename), map_location=device))
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
def run_model(model, header, recording):
    data = preprocess_data(header, recording)
    dl_scalar_pred, dl_bin_pred = model.inference(data, class_names=False, bin_pred_thr=0.5)
    # TODO: merge results from special detectors
    if len(TrainCfg.special_classes) > 0:
        pass
    else:
        labels = dl_bin_pred
        probabilities = dl_scalar_pred

    classes = []
    return classes, labels, probabilities


def preprocess_data(header:str, recording:np.ndarray):
    """
    modified from data_reader.py and dataset.py
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
    df_leads = df_leads.rename(columns={"sig_name": "lead_name", "units":"adc_units", "file_name":"filename",})
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

    # transforms performed while training
    if TrainCfg.bandpass is not None:
        data = butter_bandpass_filter(
            data,
            lowcut=TrainCfg.bandpass[0],
            highcut=TrainCfg.bandpass[1],
            order=TrainCfg.bandpass_order,
            fs=TrainCfg.fs,
        )
    # TODO: splice too long record into batches
    if TrainCfg.normalize_data:
        data = (data - np.mean(data)) / np.std(data)
    
    # add batch dimension
    data = data[np.newaxis,...].astype(np.float32)
    return data



################################################################################
#
# Other functions
#
################################################################################

# Extract features from the header and recording.
# def get_features(header, recording, leads):
#     # Extract age.
#     age = get_age(header)
#     if age is None:
#         age = float('nan')

#     # Extract sex. Encode as 0 for female, 1 for male, and NaN for other.
#     sex = get_sex(header)
#     if sex in ('Female', 'female', 'F', 'f'):
#         sex = 0
#     elif sex in ('Male', 'male', 'M', 'm'):
#         sex = 1
#     else:
#         sex = float('nan')

#     # Reorder/reselect leads in recordings.
#     available_leads = get_leads(header)
#     indices = list()
#     for lead in leads:
#         i = available_leads.index(lead)
#         indices.append(i)
#     recording = recording[indices, :]

#     # Pre-process recordings.
#     adc_gains = get_adcgains(header, leads)
#     baselines = get_baselines(header, leads)
#     num_leads = len(leads)
#     for i in range(num_leads):
#         recording[i, :] = (recording[i, :] - baselines[i]) / adc_gains[i]

#     # Compute the root mean square of each ECG lead signal.
#     rms = np.zeros(num_leads, dtype=np.float32)
#     for i in range(num_leads):
#         x = recording[i, :]
#         rms[i] = np.sqrt(np.sum(x**2) / np.size(x))

#     return age, sex, rms
